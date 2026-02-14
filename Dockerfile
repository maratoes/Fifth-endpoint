FROM runpod/pytorch:1.0.3-cu1290-torch291-ubuntu2204

RUN apt-get update && apt-get install -y git curl wget && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY handler.py .
COPY config.yaml .
COPY scripts ./scripts

ENV MODEL_NAME="browser-use/bu-30b-a3b-preview"
ENV MAX_MODEL_LEN=65536
ENV TENSOR_PARALLEL_SIZE=1
ENV GPU_MEMORY_UTILIZATION=0.90
ENV TRUST_REMOTE_CODE=True
ENV DTYPE="float16"

CMD ["python", "-u", "handler.py"]
