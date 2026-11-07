FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[dash]"
COPY configs ./configs
COPY prompts ./prompts
COPY dashboard ./dashboard
COPY data/items.jsonl ./data/items.jsonl
COPY labels ./labels
COPY runs ./runs
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
