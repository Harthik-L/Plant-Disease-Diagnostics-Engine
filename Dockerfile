# Use an official lightweight Python runtime
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose port 7860 (Hugging Face default port)
EXPOSE 7860

# Command to run the FastAPI application
CMD ["uvicorn", "src.api.py:app", "--host", "0.0.0.0", "--port", "7860"]