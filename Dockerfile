FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir \
    fastapi==0.115.12 \
    pydantic==2.11.3 \
    opencv-python==4.11.0.86 \
    pillow==11.2.1 \
    pytesseract==0.3.13 \
    python-multipart==0.0.20 \
    uvicorn==0.34.2

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]
