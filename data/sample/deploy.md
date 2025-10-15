# Deployment Notes
To deploy on Fly.io, ensure Dockerfile builds the app and exposes port 8000.
Use `fly secrets` to set OPENAI_API_KEY. For local LLMs, set OPENAI_BASE_URL accordingly.
