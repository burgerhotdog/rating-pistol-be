from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pytesseract import image_to_string
from PIL import Image
import io
from difflib import get_close_matches
from data.crops import CROPS
from data.avatar_names import AVATAR_NAMES
from data.weapon_names import WEAPON_NAMES
from data.mainstats import MAINSTATS

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
    cleaned_text = text.strip().split()[0]
    
    if cleaned_text == "Rover":
        if is_color_in_range(color, (110, 145, 138)): # Aero
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

def weapon_name_to_id(text: str) -> str:
    cleaned_text = text.strip()
    
    if cleaned_text in WEAPON_NAMES:
        return WEAPON_NAMES[cleaned_text]
    
    # If no exact match, try fuzzy matching
    matches = get_close_matches(cleaned_text, WEAPON_NAMES.keys(), n=1, cutoff=0.8)
    if matches:
        return WEAPON_NAMES[matches[0]]
    
    return "Unknown"

def mainstat_translate(text: str) -> str | None:
    cleaned_text = text.strip()
    
    if cleaned_text in MAINSTATS:
        return MAINSTATS[cleaned_text]
    
    # If no exact match, try fuzzy matching
    matches = get_close_matches(cleaned_text, MAINSTATS.keys(), n=1, cutoff=0.8)
    if matches:
        return MAINSTATS[matches[0]]
    
    return None

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
    avatar_name = image_to_string(image.crop(CROPS["avatar_name"]))
    avatar_color = get_average_color(image.crop(CROPS["avatar_color"]))
    weapon_name = image_to_string(image.crop(CROPS["weapon_name"]))
    echo0_main = image_to_string(image.crop(CROPS["echo0_main"]))
    echo1_main = image_to_string(image.crop(CROPS["echo1_main"]))
    echo2_main = image_to_string(image.crop(CROPS["echo2_main"]))
    echo3_main = image_to_string(image.crop(CROPS["echo3_main"]))
    echo4_main = image_to_string(image.crop(CROPS["echo4_main"]))
    
    return JSONResponse(content={
        "data": {
            "id": avatar_name_to_id(avatar_name, avatar_color),
            "level": 90,
            "weaponId": weapon_name_to_id(weapon_name),
            "weaponLevel": 90,
            "equipList": [
                {
                    "setId": "001",
                    "stat": mainstat_translate(echo0_main),
                },
                {
                    "setId": "001",
                    "stat": mainstat_translate(echo1_main),
                },
                {
                    "setId": "001",
                    "stat": mainstat_translate(echo2_main),
                },
                {
                    "setId": "001",
                    "stat": mainstat_translate(echo3_main),
                },
                {
                    "setId": "001",
                    "stat": mainstat_translate(echo4_main),
                },
            ],
        },
        "raw_data": {
            "id": avatar_name,
            "level": 90,
            "weaponId": weapon_name,
            "weaponLevel": 90,
            "equipList": [
                {
                    "setId": "001",
                    "stat": echo0_main,
                },
                {
                    "setId": "001",
                    "stat": echo1_main,
                },
                {
                    "setId": "001",
                    "stat": echo2_main,
                },
                {
                    "setId": "001",
                    "stat": echo3_main,
                },
                {
                    "setId": "001",
                    "stat": echo4_main,
                },
            ],
        },
    })
