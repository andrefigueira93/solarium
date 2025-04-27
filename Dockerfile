FROM python:3.12-slim

WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy required files
COPY pyproject.toml .
COPY main.py .
COPY .env .

# Install Python dependencies using pip directly
RUN pip install --no-cache-dir .

# Expose MCP server port
EXPOSE 8000

# Configure environment variables (can be overridden at runtime)
ENV BYBIT_API_KEY=""
ENV BYBIT_API_SECRET=""
ENV BYBIT_TESTNET="True"

# Command to start the MCP server
CMD ["mcp", "run", "main.py", "-t", "sse"] 