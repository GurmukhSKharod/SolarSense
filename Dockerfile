FROM mambaorg/micromamba:1.5.7

# Create the conda env
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml && \
    micromamba clean --all --yes

# App code
WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

ENV PYTHONPATH=/app

# IMPORTANT: use a shell so $PORT is expanded by the container runtime
CMD ["bash","-lc","micromamba run -n base python -m uvicorn scripts.backend.api:app --host 0.0.0.0 --port $PORT"]