# === Base Image ===
FROM python:3.11-slim

# === Set Working Directory ===
WORKDIR /app

# === Copy Project Files ===
COPY . /app

# === Install System Dependencies ===
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# === Install Python Dependencies ===
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir fastapi uvicorn tensorflow scikit-learn opencv-python-headless joblib numpy

# === Expose Port ===
EXPOSE 8000

# === Start FastAPI App ===
CMD ["uvicorn", "model_testing_fastAPI:app", "--host", "0.0.0.0", "--port", "8000"]
