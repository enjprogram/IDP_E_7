# AI Services Hub

A full-stack application for serving multiple AI/ML services, featuring Computer Vision and NLP capabilities. The backend leverages MLflow for model registry and experiment tracking, while the frontend provides an interactive dashboard for all services.

---

## Services

### Computer Vision
- **Species Classifier** — CNN model (TensorFlow/Keras) with versioned model registry via MLflow

### NLP Services
- **Ticket Classifier** — Fine-tuned BERT for support ticket categorisation
- **Named Entity Recognition** — Regex-based extraction of ORDER_ID, DATE, and EMAIL entities
- **Draft Response Generator** — OpenAI GPT-4o mini with template fallback

---

## Project Structure

```
# Backend — ML models, API endpoints
# Frontend — interactive dashboard
# Nginx
```

Each service runs in its own **isolated virtual environment**.

---

### Docker

## Running the Application with Docker/Nginx
```
docker-compose up --build -d
docker-compose -d mlflow tensorboard # start mlflow and tensorboard containers

docker-compose run --rm backend uv run python scripts/train.py # train nlp model
docker-compose run --rm backend uv jupyter nbconvert --to notebook --execute cnn_book.ipynb --output executed_output.ipynb # train cnn model

docker-compose up -d # launch the remaining containers

.\start.ps1
```

### Access the services

| Service | URL | Notes |
|---|---|---|
| Frontend (Streamlit) | http://localhost:80 | Main dashboard |
| Backend API (Swagger) | http://localhost:8000/docs | FastAPI interactive docs |
| Backend API (ReDoc) | http://localhost:8000/redoc | |
| Backend health check | http://localhost:8000/health | |
| MLflow UI | http://localhost:5000 | Experiment tracking & model registry |
| MLflow UI | http://localhost:5000/#/models | model registry |
| TensorBoard | http://localhost:6006 | Training logs & metrics |
