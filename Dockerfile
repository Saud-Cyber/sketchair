FROM python:3.11-slim

# System libraries required by MediaPipe (libGLESv2, EGL, etc.)
# and OpenCV (libGL, libSM, libX11 family) at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgles2-mesa \
    libegl1-mesa \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so Docker can cache this layer
# separately from your source code.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (web_server.py, Python/, web version/, etc.)
COPY . .

# Render sets $PORT at runtime; uvicorn must bind to it.
CMD ["sh", "-c", "uvicorn web_server:app --host 0.0.0.0 --port $PORT"]