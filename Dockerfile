FROM python:3.11-slim

# Install dependencies
RUN pip install --no-cache-dir \
    mlflow==2.22.1 \
    pandas==2.2.2 \
    scikit-learn==1.6.1 \
    numpy==2.0.2 \
    scipy==1.15.2 \
    cloudpickle==3.1.1 \
    psutil==7.0.0

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Expose MLflow port
EXPOSE 5000

# Command to run MLflow server or model serving
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]