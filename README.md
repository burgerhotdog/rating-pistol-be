## Setup

1. Build the image:
```bash
docker build -t rating-pistol-be .
```

2. Run the container:
```bash
docker run -p 8000:80 -v $(pwd):/app rating-pistol-be
```

## Testing the API

You can test the OCR endpoint using curl or Postman:

```bash
curl -X POST -F "file=@path/to/your/image.jpg" http://localhost:8000/ocr/
```

Or visit http://localhost:8000/docs for the interactive Swagger documentation.

## CORS config

Accepts requests from:
- https://burgerhotdog.github.io/rating-pistol/
- http://localhost:5173/rating-pistol/
