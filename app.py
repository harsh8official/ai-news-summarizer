import os
import re
from functools import lru_cache

import nltk
import requests
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

try:
    from transformers import pipeline
except ImportError:
    pipeline = None


app = Flask(__name__)

# Replace this value with your own key from https://newsapi.org/
# You can also set NEWS_API_KEY in your environment for deployment.
API_KEY = os.getenv("NEWS_API_KEY", "7853bebe63214572a496e840ba8467c5")
NEWS_API_URL = "https://newsapi.org/v2/top-headlines"


def ensure_nltk_data():
    """Download small NLTK resources only if they are missing."""
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
    }

    for path, package in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


@lru_cache(maxsize=1)
def get_summarizer():
    """Load a summarizer once, then reuse it for all requests.

    Some local transformer installs do not expose the "summarization" task.
    The app still tries the required pipeline first, then uses a manual
    HuggingFace model fallback, and finally falls back to NLTK summarization.
    """
    model_name = "sshleifer/distilbart-cnn-12-6"

    if pipeline is None:
        return {"type": "nltk_extractive"}

    try:
        return {
            "type": "pipeline",
            "model": pipeline("summarization", model=model_name),
        }
    except Exception:
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            return {
                "type": "manual_transformer",
                "tokenizer": tokenizer,
                "model": model,
            }
        except Exception:
            return {"type": "nltk_extractive"}


def clean_text(text):
    """Basic cleanup before summarization."""
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def preprocess_text(text):
    """Run beginner-friendly NLP preprocessing with NLTK."""
    ensure_nltk_data()
    text = clean_text(text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words("english"))

    filtered_tokens = [
        word.lower()
        for word in words
        if word.isalpha() and word.lower() not in stop_words
    ]

    return {
        "sentences": sent_tokenize(text),
        "tokens": words,
        "filtered_tokens": filtered_tokens,
    }


def split_text_for_model(text, max_words=420):
    """Keep input small enough for common summarization models."""
    words = clean_text(text).split()
    return " ".join(words[:max_words])


def extractive_summary(text, max_sentences=2):
    """Create a simple NLTK frequency-based summary as a safe fallback."""
    ensure_nltk_data()
    text = clean_text(text)
    sentences = sent_tokenize(text)

    if len(sentences) <= max_sentences:
        return text

    stop_words = set(stopwords.words("english"))
    words = [
        word.lower()
        for word in word_tokenize(text)
        if word.isalpha() and word.lower() not in stop_words
    ]

    if not words:
        return " ".join(sentences[:max_sentences])

    word_frequency = {}
    for word in words:
        word_frequency[word] = word_frequency.get(word, 0) + 1

    sentence_scores = []
    for index, sentence in enumerate(sentences):
        score = 0
        for word in word_tokenize(sentence.lower()):
            score += word_frequency.get(word, 0)
        sentence_scores.append((score, index, sentence))

    best_sentences = sorted(sentence_scores, reverse=True)[:max_sentences]
    best_sentences = sorted(best_sentences, key=lambda item: item[1])
    return " ".join(sentence for _, _, sentence in best_sentences)


def summarize_text(text):
    """Generate an AI summary using HuggingFace transformers."""
    text = split_text_for_model(text)

    if len(text.split()) < 25:
        return text or "No content available for summarization."

    summarizer = get_summarizer()

    if summarizer["type"] == "pipeline":
        summary = summarizer["model"](
            text,
            max_length=40,
            min_length=10,
            do_sample=False,
        )
        return summary[0]["summary_text"]

    if summarizer["type"] == "manual_transformer":
        tokenizer = summarizer["tokenizer"]
        model = summarizer["model"]
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        output_ids = model.generate(
            **inputs,
            max_length=40,
            min_length=10,
            do_sample=False,
            num_beams=4,
        )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return extractive_summary(text)


def fetch_news(topic="", category="technology"):
    """Fetch latest headlines from NewsAPI."""
    if not API_KEY or API_KEY == "YOUR_NEWS_API_KEY":
        return [], "Add your NewsAPI key in app.py or set NEWS_API_KEY to fetch live news."

    params = {
        "apiKey": API_KEY,
        "country": "us",
        "pageSize": 8,
        "category": category,
    }

    if topic:
        params.pop("category", None)
        params["q"] = topic

    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok":
            return [], data.get("message", "NewsAPI returned an error.")

        return data.get("articles", []), None
    except requests.RequestException as error:
        return [], f"Could not fetch news right now: {error}"


def build_news_cards(articles):
    """Summarize descriptions and prepare cards for the template."""
    cards = []

    for article in articles:
        title = article.get("title") or "Untitled article"
        description = article.get("description") or article.get("content") or ""
        article_url = article.get("url") or "#"
        image_url = article.get("urlToImage")
        source = (article.get("source") or {}).get("name", "News source")

        try:
            summary = summarize_text(description) if description else "Summary not available."
        except Exception as error:
            summary = f"Summary could not be generated: {error}"

        cards.append(
            {
                "title": title,
                "description": description,
                "summary": summary,
                "url": article_url,
                "image": image_url,
                "source": source,
            }
        )

    return cards


@app.route("/", methods=["GET", "POST"])
def index():
    topic = request.form.get("topic", "").strip()
    category = request.form.get("category", "technology")
    custom_text = request.form.get("custom_text", "").strip()
    custom_summary = ""
    nlp_stats = None
    error = None

    if request.method == "POST" and custom_text:
        try:
            nlp_data = preprocess_text(custom_text)
            custom_summary = summarize_text(custom_text)
            nlp_stats = {
                "sentence_count": len(nlp_data["sentences"]),
                "token_count": len(nlp_data["tokens"]),
                "keyword_preview": ", ".join(nlp_data["filtered_tokens"][:15]),
            }
        except Exception as exc:
            error = f"Could not summarize custom text: {exc}"

    articles, news_error = fetch_news(topic=topic, category=category)
    news_cards = build_news_cards(articles) if articles else []

    if news_error:
        error = news_error if not error else f"{error} {news_error}"

    return render_template(
        "index.html",
        news_cards=news_cards,
        custom_summary=custom_summary,
        custom_text=custom_text,
        nlp_stats=nlp_stats,
        topic=topic,
        category=category,
        error=error,
        categories=["technology", "sports", "business", "health"],
    )


if __name__ == "__main__":
    app.run(debug=True)
