FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src HF_HOME=/models
# CPU-only torch (much smaller than the CUDA build) for the bge reranker
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src src
COPY api api
COPY app app
COPY scripts scripts
COPY data/kb data/kb
COPY data/eval data/eval
EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
