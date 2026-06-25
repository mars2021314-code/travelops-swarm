#!/usr/bin/env bash

set -euo pipefail

echo "Starting TravelOps Swarm deployment..."

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and add your key."
    exit 1
fi

mkdir -p customer_support_chat/data/qdrant_storage
mkdir -p logs

echo "Stopping existing services..."
docker compose down || true

echo "Building and starting services..."
docker compose up --build -d

echo "Waiting for services..."
sleep 30

echo "Checking health..."
for i in {1..10}; do
    if curl -f http://localhost/health > /dev/null 2>&1; then
        echo "Service started successfully."
        break
    fi
    echo "Waiting for service startup... ($i/10)"
    sleep 10
done

if [ "$i" -eq 10 ]; then
    echo "Service startup failed."
    docker compose logs
    exit 1
fi

echo "Running performance monitor..."
python performance_monitor.py

echo "Deployment complete."
echo "API: http://localhost"
echo "Health: http://localhost/health"
echo "Metrics: http://localhost/metrics"
echo "Logs: docker compose logs -f"
echo "Stop: docker compose down"
