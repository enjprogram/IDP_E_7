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

## Training the Models

Both models are pre-trained and ready to use. To retrain:

**NLP (BERT) model:**
```bash
cd fastapi
uv run python scripts/train.py
```

**CNN model:**
```bash
cd fastapi
jupyter nbconvert --to notebook --execute cnn_book.ipynb --output executed_output.ipynb
```


## Running the Application without Docker locally

Open four terminal windows and run each service:

### 1 — Backend API
```bash
cd fastapi
.venv\Scripts\activate
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

API available at:
- Swagger UI → http://127.0.0.1:8000/docs
- ReDoc → http://127.0.0.1:8000/redoc
- Health check → http://127.0.0.1:8000/health

### 2 — Frontend Dashboard
```bash
cd streamlitapi
.venv\Scripts\activate
streamlit run streamlit_app.py
```

Dashboard available at → http://localhost:8502

### 3 — TensorBoard
```bash
cd fastapi
python -m tensorboard.main --logdir logs/
```

TensorBoard available at → http://localhost:6006

### 4 — MLflow UI
```bash
cd fastapi
uv run mlflow ui --backend-store-uri mlflows_runs/
```

MLflow available at → http://localhost:5000

### Docker

# Build and start
docker-compose up --build -d
# Stop
docker-compose down
# Logs
docker-compose logs -f
# Status
docker-compose ps

# Collect logs
docker-compose logs backend > backend_logs.txt


## Running the Application with Docker

## Running the Application with Docker/Nginx
docker-compose up --build -d
.\start.ps1
