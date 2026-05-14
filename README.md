# AI Summary Studio

A Flask-based NLP summarizer that can summarize pasted text, uploaded PDF files, and YouTube video transcripts.

## Features

- Text summarization from pasted content
- PDF upload and summary generation
- YouTube link summarization using video transcripts
- NLTK preprocessing and fallback summarization
- Optional HuggingFace transformer summarization for local/high-memory use
- Responsive Bootstrap UI
- Dark mode toggle
- Render-ready lightweight deployment setup

## Run Locally

```powershell
cd D:\ai_news_summ
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Extra Packages

The new PDF and YouTube features use:

```powershell
pip install pypdf youtube-transcript-api
```

## Render Deployment Notes

Set these environment variables on Render:

```text
NEWS_API_KEY=your_newsapi_key_here
ENABLE_TRANSFORMERS=false
```

Use this build command:

```text
pip install -r requirements-render.txt
```

Use this start command:

```text
gunicorn --bind 0.0.0.0:$PORT app:app
```

`ENABLE_TRANSFORMERS=false` is recommended on free deployment platforms because `torch` and HuggingFace transformer loading can cause worker timeouts or memory errors. The app will still summarize using the NLTK fallback. Use `ENABLE_TRANSFORMERS=true` only on a server with enough memory.

## Notes

- PDF summaries work best with text-based PDFs, not scanned image PDFs.
- YouTube summaries require the video to have captions/transcripts available.
- Some YouTube videos may block transcript access depending on region, language, or caption settings.
