FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf \
    NIRNAY_HOST=0.0.0.0 \
    NIRNAY_PORT=8000

WORKDIR /app

RUN pip install --no-cache-dir nirnay-450m

RUN mkdir -p /data/hf
VOLUME ["/data/hf"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=5)"

# The checkpoint downloads once into /data/hf, then the server runs offline.
CMD ["nirnay", "serve", "--host", "0.0.0.0", "--port", "8000"]
