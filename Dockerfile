FROM python:3.12-slim

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    MODULE_NAME=app.main \
    APP_HOST=0.0.0.0

RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=appuser:appgroup prediction_model/ /app/prediction_model/

COPY --chown=appuser:appgroup app/ /app/app/

RUN chown -R appuser:appgroup /app

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

USER appuser

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "app.main:app"]
