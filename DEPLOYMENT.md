# Netoval Professional - Deployment Guide

## Architecture Overview

Netoval Professional is a production-grade AI badminton analysis platform with:
- Advanced court detection (multiple strategies)
- Real-time player skeleton tracking
- Deep biomechanical analysis
- Live video streaming analysis
- Professional metrics and reporting

## System Requirements

### Hardware Minimum
- **GPU**: NVIDIA GPU with 4GB VRAM (RTX 3050 or better recommended)
- **CPU**: 8-core processor (Intel i7/AMD Ryzen 7 or better)
- **RAM**: 16GB
- **Storage**: 100GB SSD for models and video cache

### Hardware Recommended
- **GPU**: NVIDIA RTX 3090 or RTX 4090 (24GB VRAM)
- **CPU**: 16+ core processor
- **RAM**: 32GB+
- **Storage**: 500GB+ NVMe SSD

### Software Requirements
- Python 3.9+
- CUDA 11.8+
- cuDNN 8.6+
- FFmpeg (for video processing)

## Installation

### 1. Clone and Setup

```bash
git clone <repository-url>
cd netoval
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements_professional.txt
```

### 3. Download Models

Models are auto-downloaded on first use, but you can pre-download:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8s-pose.pt'); YOLO('yolov8n.pt')"
```

### 4. Environment Configuration

Create `.env` file:

```env
# Flask
FLASK_ENV=production
FLASK_APP=app_professional.py

# Redis (for caching/queuing)
REDIS_URL=redis://localhost:6379/0

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/netoval

# Security
SECRET_KEY=your-secret-key-here

# Video Processing
MAX_VIDEO_LENGTH_SEC=300
MAX_VIDEO_SIZE_MB=512
UPLOAD_FOLDER=./uploads

# GPU
USE_CUDA=1
CUDA_VISIBLE_DEVICES=0
```

## Running Locally

### Development Mode

```bash
python app_professional.py
```

Server runs at `http://localhost:5000`

### Production Mode

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app_professional:app
```

## Docker Deployment

### Build Image

```bash
docker build -f Dockerfile.professional -t netoval:latest .
```

### Run Container

```bash
docker run -d \
  --gpus all \
  -p 5000:5000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/models:/app/models \
  --env-file .env \
  --name netoval \
  netoval:latest
```

### Docker Compose

```bash
docker-compose -f docker-compose.professional.yml up -d
```

## API Reference

### Upload and Detect Players

**POST** `/detect-players`
- Accepts: Video file
- Returns: Detected players, frame preview, court calibration status

```bash
curl -X POST -F "video=@match.mp4" http://localhost:5000/detect-players
```

### Calibrate Court

**POST** `/calibrate-court`
- Accepts: Corner points [[x1,y1], [x2,y2], ...]
- Returns: Calibration success, diagnostics

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"corners": [[100,100], [500,100], [500,400], [100,400]]}' \
  http://localhost:5000/calibrate-court
```

### Live Analysis Stream

**POST** `/analyze-live`
- Accepts: player_id, skip_frames
- Returns: Server-Sent Events stream of frame analysis

```javascript
const eventSource = new EventSource('/analyze-live');
eventSource.onmessage = (event) => {
  const frameData = JSON.parse(event.data);
  console.log(frameData);
};
```

### Player Report

**GET** `/player-report/<player_id>`
- Returns: Comprehensive analysis report

```bash
curl http://localhost:5000/player-report/0
```

### Court Diagnostics

**GET** `/court-diagnostics`
- Returns: Current court calibration info

```bash
curl http://localhost:5000/court-diagnostics
```

## Performance Tuning

### GPU Optimization

```python
# In app_professional.py
pose_model.to("cuda")  # Uses GPU
torch.set_float32_matmul_precision('high')  # Faster computation
```

### Frame Skipping

For faster analysis, skip frames:

```bash
POST /analyze-live
{
  "player_id": 0,
  "skip_frames": 2  # Analyze every 3rd frame
}
```

### Batch Processing

Enable multi-threaded frame processing:

```python
from live_video_analyzer import LiveVideoAnalyzer
analyzer = LiveVideoAnalyzer(
  pose_model, yolo_model, 
  max_queue_size=60  # Increase queue for faster processing
)
```

## Monitoring

### Health Check

```bash
curl http://localhost:5000/health
```

### Logging

All activities logged to `netoval.log`:

```bash
tail -f netoval.log
```

### Performance Metrics

Enable Prometheus metrics (optional):

```python
from prometheus_client import Counter, Histogram

video_processed = Counter('videos_processed_total', 'Total videos processed')
analysis_time = Histogram('analysis_time_seconds', 'Analysis time')
```

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch size or use CPU
torch.cuda.empty_cache()
device = "cpu"  # Fallback
```

### FFmpeg Issues

```bash
# On Linux
sudo apt-get install ffmpeg

# On macOS
brew install ffmpeg

# On Windows
choco install ffmpeg
```

### Video Not Processing

Check logs:
```bash
tail -f netoval.log | grep ERROR
```

Common issues:
- Video codec not supported (convert to H.264)
- Insufficient disk space
- Model not downloaded

## Production Checklist

- [ ] HTTPS enabled (nginx SSL)
- [ ] Rate limiting configured
- [ ] Database backups scheduled
- [ ] GPU monitoring enabled
- [ ] Disk space monitoring (>20% free)
- [ ] Log rotation configured
- [ ] Database indexes created
- [ ] CDN for static assets (optional)
- [ ] Monitoring alerts set up
- [ ] Security audit completed

## Advanced Configuration

### Using Custom Models

```python
# In app_professional.py
yolo_model = YOLO("/path/to/custom/model.pt")
pose_model = YOLO("/path/to/custom/pose/model.pt")
```

### Multi-GPU Setup

```python
# Use multiple GPUs
torch.nn.DataParallel(model)
```

### Distributed Processing

For large scale deployments:

```python
from celery import Celery

celery = Celery('netoval')
celery.conf.update(
  broker='redis://redis:6379/0',
  backend='redis://redis:6379/0'
)

@celery.task
def analyze_video(video_path):
    # Long-running analysis
    pass
```

## Support & Troubleshooting

### Issue: Model download fails

```bash
# Manual download
cd models/
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt
```

### Issue: Court detection fails

Try manual calibration via `/calibrate-court` endpoint.

### Issue: High latency

- Increase skip_frames for faster processing
- Check GPU utilization: `nvidia-smi`
- Consider adding more GPU memory
- Use faster model variant (yolov8n instead of yolov8s)

## Scaling to Production

1. **Load Balancing**: Use Nginx to distribute requests
2. **Video Queue**: Use Redis for job queuing
3. **Database**: PostgreSQL for storing analysis results
4. **Caching**: Redis for frequent queries
5. **CDN**: CloudFront/Cloudflare for static files
6. **Monitoring**: Prometheus + Grafana
7. **Logging**: ELK stack (Elasticsearch, Logstash, Kibana)

## License & Attribution

Netoval Professional uses:
- YOLOv8 (Ultralytics)
- PyTorch (Meta)
- OpenCV (BSD License)

See LICENSE.md for details.
