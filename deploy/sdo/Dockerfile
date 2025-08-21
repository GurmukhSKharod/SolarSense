FROM mambaorg/micromamba:1.5.7

COPY --chown=mambauser:mambauser deploy/sdo/environment.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml && micromamba clean --all --yes

WORKDIR /app
COPY --chown=mambauser:mambauser . /app

ENV PYTHONUNBUFFERED=1
# allow SunPy path
ENV NO_SUNPY=0
ENV UVICORN_WORKERS=1

CMD ["micromamba","run","-n","base","python","-m","uvicorn","scripts.backend.api:app","--host","0.0.0.0","--port","${PORT:-8000}"]
