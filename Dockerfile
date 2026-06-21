FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY requirements.app.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.app.txt

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin semvision

COPY --chown=semvision:semvision app.py config.py ./
COPY --chown=semvision:semvision src/__init__.py src/config_loader.py ./src/
COPY --chown=semvision:semvision src/ui/ ./src/ui/
COPY --chown=semvision:semvision src/visualization/ ./src/visualization/
COPY --chown=semvision:semvision .streamlit/ ./.streamlit/
COPY --chown=semvision:semvision data/demo/demo_samples.pkl ./data/demo/demo_samples.pkl
COPY --chown=semvision:semvision artifacts/reports/*.json ./artifacts/reports/

USER semvision

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3).read()" || exit 1

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
