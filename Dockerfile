FROM mambaorg/micromamba:1.5.7

# Create env
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml && \
    micromamba clean --all --yes

WORKDIR /app
COPY . /app

ENV PYTHONPATH=/app
ENV PORT=8000

# Run FastAPI
CMD ["bash","-lc","uvicorn scripts.backend.api:app --host 0.0.0.0 --port ${PORT}"]
