# syntax=docker/dockerfile:1
FROM python:3.12.14-slim-bookworm
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NUTRI_DB=/app/local-data/nutriscope.sqlite
COPY requirements.lock ./
RUN --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then PIP_CERT=/run/secrets/proxy_ca pip install --no-cache-dir -r requirements.lock; else pip install --no-cache-dir -r requirements.lock; fi
RUN useradd --uid 10001 --create-home nutri && mkdir /app/local-data && chown nutri:nutri /app/local-data
COPY --chown=nutri:nutri app ./app
COPY --chown=nutri:nutri data/foods.csv data/recipes.json ./data/
COPY --chown=nutri:nutri config.yaml ./
RUN mkdir /app/models && chown nutri:nutri /app/models
USER nutri
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
