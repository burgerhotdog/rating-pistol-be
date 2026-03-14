from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

from app.services.ocr_service import process_image

router = APIRouter()


@router.post("/ocr/")
async def ocr(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    result = process_image(image)

    if result is None:
        return JSONResponse(status_code=400, content={"error": "Could not detect character name region"})

    if "error" in result:
        return JSONResponse(status_code=400, content=result)

    return JSONResponse(content=result)
