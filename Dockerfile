# NOTE: Keep this tag in sync with what actually exists on Docker Hub.
# The previously referenced 2.1.0 tag is not published in runpod/pytorch.
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

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
