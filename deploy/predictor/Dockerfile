FROM mambaorg/micromamba:1.5.7

# Create env
COPY --chown=mambauser:mambauser deploy/predictor/environment.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml && micromamba clean --all --yes

# Install CPU PyTorch (correct way; note no +cpu in the version)
RUN micromamba run -n base python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2

# App
WORKDIR /app
COPY . /app
ENV PYTHONPATH=/app
ENV PORT=10000
EXPOSE 10000

# keep memory low: single worker; NO_SUNPY disables heavy bits in sdo.py
ENV WEB_CONCURRENCY=1
ENV NO_SUNPY=1

CMD micromamba run -n base python -m uvicorn scripts.backend.api:app \
    --host 0.0.0.0 --port ${PORT:-10000} --workers 1
