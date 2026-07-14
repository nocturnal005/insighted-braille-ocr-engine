FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY samples ./samples
COPY limitations.md README.md ./

RUN useradd --create-home ocr
USER ocr

EXPOSE 8000

# Managed hosts inject $PORT and require the service to bind to it; default to
# 8000 so local `docker run` and the demo flow are unchanged.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
