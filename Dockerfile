# ModelFloat MVP Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY ./src ./src
COPY README.md .
COPY SUMMARY.md .
COPY PRD.md .
COPY milestones.md .
COPY tasks.md .
COPY requirements.txt .
COPY pyproject.toml .
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install fastapi uvicorn pytest python-multipart httpx ipfshttpclient mega.py google-api-python-client google-auth-httplib2 google-auth-oauthlib

# Expose API port
EXPOSE 8000

# Default: run API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
