#!/bin/bash

# ParkRun Predictor - Docker Update Script
echo "🔄 Updating ParkRun Predictor Docker container..."

# Stop any running containers
echo "🛑 Stopping running containers..."
docker stop $(docker ps -q --filter ancestor=parkrun-predictor) 2>/dev/null || true

# Remove old container
echo "🗑️  Removing old container..."
docker rm $(docker ps -aq --filter ancestor=parkrun-predictor) 2>/dev/null || true

# Rebuild the image
echo "🔨 Rebuilding Docker image..."
docker build -t parkrun-predictor .

if [ $? -ne 0 ]; then
    echo "❌ Failed to build Docker image"
    exit 1
fi

echo "✅ Docker image rebuilt successfully"

# Find available port
PORT=8501
while lsof -i :$PORT > /dev/null 2>&1; do
    PORT=$((PORT + 1))
    if [ $PORT -gt 8600 ]; then
        echo "❌ No available ports found in range 8501-8600"
        exit 1
    fi
done

echo "🔍 Using port: $PORT"

# Run the updated container
echo "🚀 Starting updated ParkRun Predictor container..."
echo "📱 Access the app at: http://localhost:$PORT"
echo "🛑 Press Ctrl+C to stop the container"
echo ""

docker run --rm -p $PORT:8501 \
    -v "$(pwd)/output:/app/output" \
    -v "$(pwd)/models:/app/models" \
    parkrun-predictor
