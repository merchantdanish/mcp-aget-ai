FROM python:3.10-slim

WORKDIR /app

# Install essential packages
RUN apt-get update && apt-get install -y git curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install pip dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY mcp_agent.config.yaml .
COPY main.py .

# Create output directory
RUN mkdir -p /app/output

# Run the MCP application
CMD ["python", "main.py"]