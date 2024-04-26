FROM python:3.11-buster as builder

RUN pip install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app/
ADD pyproject.toml poetry.lock ./
# Streamlit app needs a lot of dependencies

RUN \
    --mount=type=cache,target=$POETRY_CACHE_DIR \
    poetry install \
    --no-root \
    --no-ansi \
    --with streamlit-app,ai \
    --without dev

FROM python:3.11-slim-buster as embeddings
ARG EMBEDDING_MODEL
ARG SENTENCE_TRANSFORMERS_HOME

WORKDIR /app/

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('${EMBEDDING_MODEL}', cache_folder='${SENTENCE_TRANSFORMERS_HOME}');"

FROM embeddings as runtime

RUN apt-get update
RUN apt-get install -y libgl-dev libmagic-dev inetutils-ping poppler-utils tesseract-ocr

WORKDIR /app/

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

ADD redbox/ /app/redbox
ADD streamlit_app/ /app/streamlit_app

ENV PYTHONPATH "/app"

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "--server.address", "0.0.0.0", "--server.port", "8501", "streamlit_app/Welcome.py" ]
