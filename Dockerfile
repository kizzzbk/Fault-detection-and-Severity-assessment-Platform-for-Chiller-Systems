# Chiller Twin web demo — runs 24/7 on any Docker host (Hugging Face Spaces, Render, Fly.io, a VPS…)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=7860
WORKDIR /app

COPY webapp/requirements.txt webapp/requirements.txt
RUN pip install --no-cache-dir -r webapp/requirements.txt

# only what inference needs
COPY src/chiller_fault_detection src/chiller_fault_detection
COPY demo/__init__.py demo/cnn_schema.py demo/pipeline.py demo/
COPY demo/models demo/models
COPY webapp webapp

RUN useradd -m app && chown -R app /app
USER app
EXPOSE 7860
HEALTHCHECK --interval=60s --timeout=5s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz')"
CMD ["sh", "-c", "uvicorn webapp.server:app --host 0.0.0.0 --port ${PORT} --workers 1"]
