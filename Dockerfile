#  base image with CUDA + PyTorch 
    FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

    # 1. System deps (optional but keeps wheels small)
    RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl && \
        rm -rf /var/lib/apt/lists/*
    
    # 2. Copy requirement list & install
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    
    # 3. Copy application code
    COPY app ./app
    #COPY .env .env             # optional: if you want .env inside the image
    
    # 4. Expose port & default cmd
    EXPOSE 8000
    CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
    