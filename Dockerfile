FROM python:3.11-slim

WORKDIR /app

# System deps (opencv often needs these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python deps
COPY requirements.txt /app/requirements.txt
RUN pip install -U pip && pip install -r requirements.txt

# Copy project
COPY . /app

# Default command: run pipeline
CMD ["python", "-m", "app", "run", "--input", "data", "--out", "outputs", "--config", "configs/default.yaml"]
