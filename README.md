# ai-news-summarizer

## Render deployment notes

Set these environment variables on Render:

```text
NEWS_API_KEY=your_newsapi_key_here
ENABLE_TRANSFORMERS=false
```

Use this start command:

```text
gunicorn --bind 0.0.0.0:$PORT app:app
```

`ENABLE_TRANSFORMERS=false` is recommended on free deployment platforms because `torch` and HuggingFace transformer loading can cause worker timeouts or memory errors. The app will still summarize using the NLTK fallback. Use `ENABLE_TRANSFORMERS=true` only on a server with enough memory.
