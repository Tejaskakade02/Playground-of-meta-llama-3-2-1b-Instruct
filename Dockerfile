FROM nvcr.io/nvidia/tritonserver:24.09-vllm-python-py3

WORKDIR /workspace

# COPY model_repository /models
# COPY vllm_workspace /workspace
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8000 8001 8002

CMD ["tritonserver", "--model-repository=/models"]
