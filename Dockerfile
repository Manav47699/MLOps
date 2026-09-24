# Use official lightweight Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies if needed (e.g., build-essential, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose ports: 8000 (FastAPI backend) and 8501 (Streamlit UI)
EXPOSE 8000 8501

# Default command runs the backend (overridden in docker-compose for UI)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
