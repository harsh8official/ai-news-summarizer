import os
import re
from functools import lru_cache

import nltk
import requests
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

# Replace this value with your own key from https://newsapi.org/
# You can also set NEWS_API_KEY in your environment for deployment.
API_KEY = os.getenv("NEWS_API_KEY", "7853bebe63214572a496e840ba8467c5")
NEWS_API_URL = "https://newsapi.org/v2/top-headlines"
ENABLE_TRANSFORMERS = os.getenv("ENABLE_TRANSFORMERS", "false").lower() == "true"


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

    if not ENABLE_TRANSFORMERS:
        return {"type": "nltk_extractive"}

    try:
        from transformers import pipeline

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


@app.route("/health")
def health():
    """Simple health check for deployment platforms."""
    return {"status": "ok"}


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


def extract_pdf_text(pdf_file):
    """Extract readable text from an uploaded PDF file."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return "", "PDF support is missing. Install it with: pip install pypdf"

    try:
        reader = PdfReader(pdf_file)
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        text = clean_text(" ".join(pages))

        if not text:
            return "", "No selectable text found in this PDF."

        return text, None
    except Exception as error:
        return "", f"Could not read this PDF: {error}"


def extract_youtube_video_id(url):
    """Extract the video id from common YouTube URL formats."""
    patterns = [
        r"(?:youtube\.com/watch\?v=)([^&]+)",
        r"(?:youtu\.be/)([^?&]+)",
        r"(?:youtube\.com/embed/)([^?&]+)",
        r"(?:youtube\.com/shorts/)([^?&]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return ""


def split_for_translation(text, max_chars=3500):
    """Split long transcript text into translation-friendly chunks."""
    words = clean_text(text).split()
    chunks = []
    current = []
    current_length = 0

    for word in words:
        next_length = current_length + len(word) + 1
        if current and next_length > max_chars:
            chunks.append(" ".join(current))
            current = [word]
            current_length = len(word)
        else:
            current.append(word)
            current_length = next_length

    if current:
        chunks.append(" ".join(current))

    return chunks


@lru_cache(maxsize=1)
def get_translation_model():
    """Load an optional local Hindi-to-English translator."""
    if not ENABLE_TRANSFORMERS:
        return None

    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_name = "Helsinki-NLP/opus-mt-hi-en"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return {"tokenizer": tokenizer, "model": model}
    except Exception:
        return None


def translate_with_local_model(text):
    """Translate with a local HuggingFace model when enabled."""
    translator = get_translation_model()
    if not translator:
        return text, "Local Hindi-to-English model is not enabled or could not be loaded."

    try:
        tokenizer = translator["tokenizer"]
        model = translator["model"]
        translated_chunks = []

        for chunk in split_for_translation(text, max_chars=1200):
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
            output_ids = model.generate(**inputs, max_length=512)
            translated_chunks.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))

        return clean_text(" ".join(translated_chunks)), None
    except Exception as error:
        return text, f"Local Hindi-to-English translation failed: {error}"


def translate_to_english(text):
    """Translate Hindi/non-English transcript text to English."""
    errors = []

    try:
        from deep_translator import GoogleTranslator, MyMemoryTranslator
    except ImportError:
        translated_text, model_error = translate_with_local_model(text)
        if not model_error:
            return translated_text, None
        return text, "Install deep-translator to translate non-English text."

    try:
        translator = GoogleTranslator(source="auto", target="en")
        translated_chunks = [
            translator.translate(chunk)
            for chunk in split_for_translation(text)
            if chunk
        ]
        return clean_text(" ".join(translated_chunks)), None
    except Exception as error:
        errors.append(f"Google Translate failed: {error}")

    try:
        translator = MyMemoryTranslator(source="hi-IN", target="en-US")
        translated_chunks = [
            translator.translate(chunk)
            for chunk in split_for_translation(text, max_chars=450)
            if chunk
        ]
        return clean_text(" ".join(translated_chunks)), None
    except Exception as error:
        errors.append(f"MyMemory Translate failed: {error}")

    translated_text, model_error = translate_with_local_model(text)
    if not model_error:
        return translated_text, None

    errors.append(model_error)
    return text, "Could not translate transcript to English. " + " ".join(errors)


def should_translate_to_english(text):
    """Detect common non-English scripts that should be translated first."""
    return bool(re.search(r"[\u0900-\u097F]", text or ""))


def prepare_english_summary_source(text):
    """Translate Hindi/Devanagari text to English before summarizing."""
    text = clean_text(text)

    if not should_translate_to_english(text):
        return text, None

    translated_text, translate_error = translate_to_english(text)
    if translate_error:
        return text, translate_error

    return translated_text, None


def translate_transcript_to_english(transcript_info):
    """Try YouTube's own transcript translation before external translation."""
    try:
        translated_transcript = transcript_info.translate("en").fetch()
        text = " ".join(item.text for item in translated_transcript)
        return clean_text(text), None
    except Exception:
        return "", "YouTube translation is not available for this transcript."


def extract_youtube_transcript(url):
    """Fetch transcript text for a YouTube video link."""
    video_id = extract_youtube_video_id(url)

    if not video_id:
        return "", "Please enter a valid YouTube video URL."

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return "", "YouTube support is missing. Install it with: pip install youtube-transcript-api"

    try:
        transcript_api = YouTubeTranscriptApi()
        transcript_list = transcript_api.list(video_id)
        transcript_info = None

        try:
            transcript_info = transcript_list.find_transcript(["en"])
        except Exception:
            transcript_info = next(iter(transcript_list), None)

        if transcript_info is None:
            return "", "No transcript is available for this video."

        transcript = transcript_info.fetch()
        text = " ".join(item.text for item in transcript)
        text = clean_text(text)
        language_code = getattr(transcript_info, "language_code", "")

        if language_code and not language_code.lower().startswith("en"):
            youtube_translation, youtube_error = translate_transcript_to_english(transcript_info)
            if youtube_translation:
                return youtube_translation, None

            translated_text, translate_error = translate_to_english(text)
            if translate_error:
                return text, f"{youtube_error} {translate_error}"
            return translated_text, None

        return text, None
    except AttributeError:
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            try:
                transcript_info = transcript_list.find_transcript(["en"])
            except Exception:
                transcript_info = next(iter(transcript_list), None)

            if transcript_info is None:
                return "", "No transcript is available for this video."

            transcript = transcript_info.fetch()
            text_parts = []
            for item in transcript:
                if isinstance(item, dict):
                    text_parts.append(item.get("text", ""))
                else:
                    text_parts.append(getattr(item, "text", ""))
            text = " ".join(text_parts)
            text = clean_text(text)
            language_code = getattr(transcript_info, "language_code", "")

            if language_code and not language_code.lower().startswith("en"):
                youtube_translation, youtube_error = translate_transcript_to_english(transcript_info)
                if youtube_translation:
                    return youtube_translation, None

                translated_text, translate_error = translate_to_english(text)
                if translate_error:
                    return text, f"{youtube_error} {translate_error}"
                return translated_text, None

            return text, None
        except Exception as error:
            return "", f"Could not get transcript for this video: {error}"
    except Exception as error:
        return "", f"Could not get transcript for this video: {error}"


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
    if request.method == "HEAD":
        return "", 200

    action = request.form.get("action", "")
    custom_text = request.form.get("custom_text", "").strip()
    youtube_url = request.form.get("youtube_url", "").strip()
    custom_summary = ""
    pdf_summary = ""
    youtube_summary = ""
    nlp_stats = None
    error = None

    if request.method == "POST" and action == "text" and custom_text:
        try:
            nlp_data = preprocess_text(custom_text)
            summary_source, translate_error = prepare_english_summary_source(custom_text)
            custom_summary = summarize_text(summary_source)
            if translate_error:
                error = f"{translate_error} Showing summary in the original language."
            nlp_stats = {
                "sentence_count": len(nlp_data["sentences"]),
                "token_count": len(nlp_data["tokens"]),
                "keyword_preview": ", ".join(nlp_data["filtered_tokens"][:15]),
            }
        except Exception as exc:
            error = f"Could not summarize custom text: {exc}"

    if request.method == "POST" and action == "pdf":
        pdf_file = request.files.get("pdf_file")

        if not pdf_file or not pdf_file.filename:
            error = "Please upload a PDF file."
        elif not pdf_file.filename.lower().endswith(".pdf"):
            error = "Only PDF files are supported."
        else:
            pdf_text, pdf_error = extract_pdf_text(pdf_file)
            if pdf_error:
                error = pdf_error
            else:
                summary_source, translate_error = prepare_english_summary_source(pdf_text)
                pdf_summary = summarize_text(summary_source)
                if translate_error:
                    error = f"{translate_error} Showing summary in the original language."

    if request.method == "POST" and action == "youtube":
        transcript_text, transcript_error = extract_youtube_transcript(youtube_url)
        if transcript_text:
            youtube_summary = summarize_text(transcript_text)
            error = transcript_error
        elif transcript_error:
            error = transcript_error

    return render_template(
        "index.html",
        custom_summary=custom_summary,
        pdf_summary=pdf_summary,
        youtube_summary=youtube_summary,
        custom_text=custom_text,
        youtube_url=youtube_url,
        nlp_stats=nlp_stats,
        error=error,
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
