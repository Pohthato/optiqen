# Build this image with the licensed badminton weights placed in worker/weights/.
# Keeping weights in the image (or a same-region RunPod model cache) is what
# prevents a worker from downloading models during a customer request.
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /opt/netoval
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_DIR=/opt/netoval/models

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
COPY worker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY worker/handler.py worker/badminton.py /opt/netoval/
COPY worker/weights/ /opt/netoval/models/

# Fail the image build only when a deliberate production build verifies its
# private assets. The runtime independently refuses to silently download them.
CMD ["python", "/opt/netoval/handler.py"]
