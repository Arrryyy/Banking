# Base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working dir
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source code
COPY . /app/

# Run preprocessing, training, and utils during build
# RUN python scripts/preprocessing.py && \
#     python scripts/train_svm.py && \
#     python scripts/utils.py

# Expose Streamlit port
EXPOSE 8501

# Launch Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]