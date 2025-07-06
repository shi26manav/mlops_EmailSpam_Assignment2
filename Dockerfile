# Use official Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5001
EXPOSE 5001

# Run the Flask app
CMD ["python", "app.py"]

