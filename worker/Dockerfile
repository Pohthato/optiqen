# Build this image with the licensed badminton weights placed in worker/weights/.
# Keeping weights in the image (or a same-region RunPod model cache) is what
# prevents a worker from downloading models during a customer request.
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /opt/netoval
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_DIR=/opt/netoval/models

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
COPY worker/requirements.txt /tmp/requirements.txt
# Ultralytics and MediaPipe each pull in their own OpenCV, and every OpenCV package writes the same
# cv2 module: keep only the one requirements.txt names (pip check will miss the others' names).
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless \
    && pip install --no-cache-dir --no-deps "$(grep '^opencv-contrib-python-headless==' /tmp/requirements.txt | tr -d '\r')"
COPY worker/handler.py worker/badminton.py /opt/netoval/
# The rally analysis (rally_job.py runs analyse_rally.py) and everything it imports.
COPY worker/rally_job.py worker/analyse_rally.py worker/body.py worker/detect_poses.py worker/detect_shuttle.py worker/find_hits.py worker/findings.py worker/hits.py worker/lift_players.py worker/players.py worker/pressure.py worker/rally_reasoning.py worker/rally_value.py worker/rally_value.json worker/shot_names.py worker/shot_names.json worker/shots.py worker/track_camera.py worker/track_players.py worker/video.py /opt/netoval/
COPY worker/geometry/ /opt/netoval/geometry/
COPY worker/audio/ /opt/netoval/audio/
COPY worker/shuttle/ /opt/netoval/shuttle/
COPY worker/evaluation/__init__.py worker/evaluation/golden.py /opt/netoval/evaluation/
COPY worker/simulation/__init__.py worker/simulation/people.py /opt/netoval/simulation/
COPY worker/weights/ /opt/netoval/models/
# The shuttle detector and the pose model are looked for in weights/ beside the analysis.
RUN ln -s /opt/netoval/models /opt/netoval/weights

# Fail the image build only when a deliberate production build verifies its
# private assets. The runtime independently refuses to silently download them.
CMD ["python", "/opt/netoval/handler.py"]
