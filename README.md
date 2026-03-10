---
title: Sentiment Analysis API
emoji: 🎭
colorFrom: orange
colorTo: red
sdk: docker
pinned: false
app_port: 7860
---

# Sentiment Analysis API

A REST API that classifies text sentiment using RoBERTa.

## Endpoints
- `GET /health` - health check
- `POST /predict` - returns POSITIVE/NEGATIVE/NEUTRAL + confidence scores