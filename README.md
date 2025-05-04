# Rating Pistol Backend

This is the backend service for the Rating Pistol application, providing OCR functionality using Tesseract.

## Prerequisites

1. Python 3.9 or higher
2. Tesseract OCR installed on your system
   - For Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
   - For Mac: `brew install tesseract`
   - For Linux: `sudo apt-get install tesseract-ocr`

## Local Development Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
uvicorn app:app --reload
```

The server will start at http://localhost:8000

## Testing the API

You can test the OCR endpoint using curl or Postman:

```bash
curl -X POST -F "file=@path/to/your/image.jpg" http://localhost:8000/ocr/
```

Or visit http://localhost:8000/docs for the interactive Swagger documentation.

## CORS Configuration

The backend is configured to accept requests from:
- https://burgerhotdog.github.io/rating-pistol/
- http://localhost:5173/rating-pistol/

## Docker Support

To run with Docker:

1. Build the image:
```bash
docker build -t rating-pistol-be .
```

2. Run the container:
```bash
docker run -p 8000:80 rating-pistol-be
``` 