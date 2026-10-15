FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install . && rm -rf build

COPY config.toml ./
COPY eval_data ./eval_data

# Run as a normal user, not root. The database lives in /app/data; mount a volume there
# to keep it between runs. Secrets come from the environment at run time (--env-file .env).
RUN useradd --create-home triagebot && mkdir data && chown triagebot data
USER triagebot

ENTRYPOINT ["triagebot"]
CMD ["--help"]
