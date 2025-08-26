# CPU-only image
FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \ 
    build-essential cmake pkg-config     libjpeg-dev libpng-dev libtiff-dev libgtk2.0-dev     libavcodec-dev libavformat-dev libswscale-dev     && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e ".[dev]"

CMD ["bash"]
