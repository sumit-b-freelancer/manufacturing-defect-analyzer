# Stage 1: Builder
FROM python:3.9-slim as builder

WORKDIR /app

COPY requirements.txt .

# Install dependencies into a virtual environment to keep image small
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim

WORKDIR /app

# Copy virtual env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run command
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
