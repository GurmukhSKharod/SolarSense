FROM mambaorg/micromamba:1.5.7

# Conda env
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml && \
    micromamba clean --all --yes

# App
WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

# Make 'scripts' resolvable as a package from repo root
ENV PYTHONPATH=/app

# Run Uvicorn inside the conda env; expand $PORT via bash so Render can bind it
CMD ["bash","-lc","micromamba run -n base python -m uvicorn scripts.backend.api:app --host 0.0.0.0 --port $PORT"]
