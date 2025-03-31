FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn pyjwt redis python-dotenv

# Copy application code
COPY . /app/

EXPOSE 8000

# Run the server
CMD ["python", "-m", "uvicorn", "http_server:app", "--host", "0.0.0.0", "--port", "8000"]