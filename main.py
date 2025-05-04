from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pytesseract import image_to_string
from PIL import Image
import io
from difflib import get_close_matches
import json

# Character name crop coordinates (left, top, right, bottom)
AVATAR_NAME_CROP = (65, 19, 620, 84)

# Valid character names
with open('AVATAR_NAMES.json', 'r') as f:
    AVATAR_NAMES = json.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://burgerhotdog.github.io",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_average_color(image):
    image = image.convert("RGB").resize((1, 1))  # Collapse to 1x1
    return image.getpixel((0, 0))  # Get that one averaged pixel

def is_color_in_range(color, target, tolerance=20):
    r, g, b = color
    tr, tg, tb = target
    
    return (tr - tolerance <= r <= tr + tolerance and
            tg - tolerance <= g <= tg + tolerance and
            tb - tolerance <= b <= tb + tolerance)

def avatar_name_to_id(text: str, color: tuple) -> str:
    cleaned_text = text.strip().replace('\n', '').split()[0]
    
    if cleaned_text == "Rover":
        if is_color_in_range(color, (255, 255, 255)): # Aero
            return "1408"
        elif is_color_in_range(color, (137, 49, 110)): # Havoc
            return "1604"
        elif is_color_in_range(color, (130, 113, 79)): # Spectro
            return "1502"
        else:
            return "Unknown"
    
    if cleaned_text in AVATAR_NAMES:
        return AVATAR_NAMES[cleaned_text]
    
    # If no exact match, try fuzzy matching
    matches = get_close_matches(cleaned_text, AVATAR_NAMES.keys(), n=1, cutoff=0.8)
    if matches:
        return AVATAR_NAMES[matches[0]]
    
    return "Unknown"

@app.post("/ocr/")
async def ocr(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Verify image dimensions
    if image.size != (1920, 1080):
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid image dimensions. Expected 1920x1080"}
        )
    
    # Process Crops
    avatar_name = image_to_string(image.crop(AVATAR_NAME_CROP))
    avatar_color = get_average_color(image.crop((20, 30, 60, 70)))
    
    return JSONResponse(content={
        "data": {
            "id": avatar_name_to_id(avatar_name, avatar_color)
        },
        "raw_data": {
            "id": avatar_name
        },
        "color": get_average_color(image.crop((20, 30, 60, 70)))
    })