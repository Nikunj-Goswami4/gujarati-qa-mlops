FROM python:3.11.7-slim

WORKDIR /app

# Prevent Python from writing .pyc files and buffer streams
ENV PYTHONUNBUFFERED=1

# Install OS-level build systems
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-docker.txt

# Copy source code
COPY src/ src/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]