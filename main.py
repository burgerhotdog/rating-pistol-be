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
from data.substats import SUBSTATS

custom_config = r'--oem 3 --psm 7'

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

def substat_translate(text: str, has_percent: bool) -> str | None:
    cleaned_text = text.strip()
    
    if cleaned_text in SUBSTATS:
        if cleaned_text in ("HP", "ATK", "DEF"):
            if has_percent:
                return cleaned_text
            else:
                return f"_{cleaned_text}"
        return SUBSTATS[cleaned_text]
    
    # If no exact match, try fuzzy matching
    matches = get_close_matches(cleaned_text, SUBSTATS.keys(), n=1, cutoff=0.8)
    if matches:
        if matches[0] in ("HP", "ATK", "DEF"):
            if has_percent:
                return matches[0]
            else:
                return f"_{matches[0]}"
        return SUBSTATS[matches[0]]
    
    return None

def value_translate(text: str) -> tuple[str | None, bool]:
    cleaned_text = text.strip()
    has_percent = cleaned_text.endswith('%')

    if has_percent:
        cleaned_text = cleaned_text[:-1].strip()

    try:
        float(cleaned_text)
        return cleaned_text, has_percent
    except ValueError:
        return None, has_percent

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
    echo0_main = image_to_string(image.crop(CROPS["echo0_main"]), config=custom_config)
    echo0_sub0 = image_to_string(image.crop(CROPS["echo0_sub0"]), config=custom_config)
    echo0_val0 = image_to_string(image.crop(CROPS["echo0_val0"]), config=custom_config)
    converted0_sub0 = value_translate(echo0_val0)
    echo0_sub1 = image_to_string(image.crop(CROPS["echo0_sub1"]), config=custom_config)
    echo0_val1 = image_to_string(image.crop(CROPS["echo0_val1"]), config=custom_config)
    converted0_sub1 = value_translate(echo0_val1)
    echo0_sub2 = image_to_string(image.crop(CROPS["echo0_sub2"]), config=custom_config)
    echo0_val2 = image_to_string(image.crop(CROPS["echo0_val2"]), config=custom_config)
    converted0_sub2 = value_translate(echo0_val2)
    echo0_sub3 = image_to_string(image.crop(CROPS["echo0_sub3"]), config=custom_config)
    echo0_val3 = image_to_string(image.crop(CROPS["echo0_val3"]), config=custom_config)
    converted0_sub3 = value_translate(echo0_val3)
    echo0_sub4 = image_to_string(image.crop(CROPS["echo0_sub4"]), config=custom_config)
    echo0_val4 = image_to_string(image.crop(CROPS["echo0_val4"]), config=custom_config)
    converted0_sub4 = value_translate(echo0_val4)

    echo1_main = image_to_string(image.crop(CROPS["echo1_main"]), config=custom_config)
    echo1_sub0 = image_to_string(image.crop(CROPS["echo1_sub0"]), config=custom_config)
    echo1_val0 = image_to_string(image.crop(CROPS["echo1_val0"]), config=custom_config)
    converted1_sub0 = value_translate(echo1_val0)
    echo1_sub1 = image_to_string(image.crop(CROPS["echo1_sub1"]), config=custom_config)
    echo1_val1 = image_to_string(image.crop(CROPS["echo1_val1"]), config=custom_config)
    converted1_sub1 = value_translate(echo1_val1)
    echo1_sub2 = image_to_string(image.crop(CROPS["echo1_sub2"]), config=custom_config)
    echo1_val2 = image_to_string(image.crop(CROPS["echo1_val2"]), config=custom_config)
    converted1_sub2 = value_translate(echo1_val2)
    echo1_sub3 = image_to_string(image.crop(CROPS["echo1_sub3"]), config=custom_config)
    echo1_val3 = image_to_string(image.crop(CROPS["echo1_val3"]), config=custom_config)
    converted1_sub3 = value_translate(echo1_val3)
    echo1_sub4 = image_to_string(image.crop(CROPS["echo1_sub4"]), config=custom_config)
    echo1_val4 = image_to_string(image.crop(CROPS["echo1_val4"]), config=custom_config)
    converted1_sub4 = value_translate(echo1_val4)
    
    echo2_main = image_to_string(image.crop(CROPS["echo2_main"]), config=custom_config)
    echo2_sub0 = image_to_string(image.crop(CROPS["echo2_sub0"]), config=custom_config)
    echo2_val0 = image_to_string(image.crop(CROPS["echo2_val0"]), config=custom_config)
    converted2_sub0 = value_translate(echo2_val0)
    echo2_sub1 = image_to_string(image.crop(CROPS["echo2_sub1"]), config=custom_config)
    echo2_val1 = image_to_string(image.crop(CROPS["echo2_val1"]), config=custom_config)
    converted2_sub1 = value_translate(echo2_val1)
    echo2_sub2 = image_to_string(image.crop(CROPS["echo2_sub2"]), config=custom_config)
    echo2_val2 = image_to_string(image.crop(CROPS["echo2_val2"]), config=custom_config)
    converted2_sub2 = value_translate(echo2_val2)
    echo2_sub3 = image_to_string(image.crop(CROPS["echo2_sub3"]), config=custom_config)
    echo2_val3 = image_to_string(image.crop(CROPS["echo2_val3"]), config=custom_config)
    converted2_sub3 = value_translate(echo2_val3)
    echo2_sub4 = image_to_string(image.crop(CROPS["echo2_sub4"]), config=custom_config)
    echo2_val4 = image_to_string(image.crop(CROPS["echo2_val4"]), config=custom_config)
    converted2_sub4 = value_translate(echo2_val4)

    echo3_main = image_to_string(image.crop(CROPS["echo3_main"]), config=custom_config)
    echo3_sub0 = image_to_string(image.crop(CROPS["echo3_sub0"]), config=custom_config)
    echo3_val0 = image_to_string(image.crop(CROPS["echo3_val0"]), config=custom_config)
    converted3_sub0 = value_translate(echo3_val0)
    echo3_sub1 = image_to_string(image.crop(CROPS["echo3_sub1"]), config=custom_config)
    echo3_val1 = image_to_string(image.crop(CROPS["echo3_val1"]), config=custom_config)
    converted3_sub1 = value_translate(echo3_val1)
    echo3_sub2 = image_to_string(image.crop(CROPS["echo3_sub2"]), config=custom_config)
    echo3_val2 = image_to_string(image.crop(CROPS["echo3_val2"]), config=custom_config)
    converted3_sub2 = value_translate(echo3_val2)
    echo3_sub3 = image_to_string(image.crop(CROPS["echo3_sub3"]), config=custom_config)
    echo3_val3 = image_to_string(image.crop(CROPS["echo3_val3"]), config=custom_config)
    converted3_sub3 = value_translate(echo3_val3)
    echo3_sub4 = image_to_string(image.crop(CROPS["echo3_sub4"]), config=custom_config)
    echo3_val4 = image_to_string(image.crop(CROPS["echo3_val4"]), config=custom_config)
    converted3_sub4 = value_translate(echo3_val4)

    echo4_main = image_to_string(image.crop(CROPS["echo4_main"]), config=custom_config)
    echo4_sub0 = image_to_string(image.crop(CROPS["echo4_sub0"]), config=custom_config)
    echo4_val0 = image_to_string(image.crop(CROPS["echo4_val0"]), config=custom_config)
    converted4_sub0 = value_translate(echo4_val0)
    echo4_sub1 = image_to_string(image.crop(CROPS["echo4_sub1"]), config=custom_config)
    echo4_val1 = image_to_string(image.crop(CROPS["echo4_val1"]), config=custom_config)
    converted4_sub1 = value_translate(echo4_val1)
    echo4_sub2 = image_to_string(image.crop(CROPS["echo4_sub2"]), config=custom_config)
    echo4_val2 = image_to_string(image.crop(CROPS["echo4_val2"]), config=custom_config)
    converted4_sub2 = value_translate(echo4_val2)
    echo4_sub3 = image_to_string(image.crop(CROPS["echo4_sub3"]), config=custom_config)
    echo4_val3 = image_to_string(image.crop(CROPS["echo4_val3"]), config=custom_config)
    converted4_sub3 = value_translate(echo4_val3)
    echo4_sub4 = image_to_string(image.crop(CROPS["echo4_sub4"]), config=custom_config)
    echo4_val4 = image_to_string(image.crop(CROPS["echo4_val4"]), config=custom_config)
    converted4_sub4 = value_translate(echo4_val4)

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
                    "statList": [
                        {
                            "stat": substat_translate(echo0_sub0, converted0_sub0[1]),
                            "value": converted0_sub0[0],
                        },
                        {
                            "stat": substat_translate(echo0_sub1, converted0_sub1[1]),
                            "value": converted0_sub1[0],
                        },
                        {
                            "stat": substat_translate(echo0_sub2, converted0_sub2[1]),
                            "value": converted0_sub2[0],
                        },
                        {
                            "stat": substat_translate(echo0_sub3, converted0_sub3[1]),
                            "value": converted0_sub3[0],
                        },
                        {
                            "stat": substat_translate(echo0_sub4, converted0_sub4[1]),
                            "value": converted0_sub4[0],
                        },
                    ],
                },
                {
                    "setId": "001",
                    "stat": mainstat_translate(echo1_main),
                    "statList": [
                        {
                            "stat": substat_translate(echo1_sub0, converted1_sub0[1]),
                            "value": converted1_sub0[0],
                        },
                        {
                            "stat": substat_translate(echo1_sub1, converted1_sub1[1]),
                            "value": converted1_sub1[0],
                        },
                        {
                            "stat": substat_translate(echo1_sub2, converted1_sub2[1]),
                            "value": converted1_sub2[0],
                        },
                        {
                            "stat": substat_translate(echo1_sub3, converted1_sub3[1]),
                            "value": converted1_sub3[0],
                        },
                        {
                            "stat": substat_translate(echo1_sub4, converted1_sub4[1]),
                            "value": converted1_sub4[0],
                        },
                    ],
                },
                {
                    "setId": "001",
                    "stat": mainstat_translate(echo2_main),
                    "statList": [
                        {
                            "stat": substat_translate(echo2_sub0, converted2_sub0[1]),
                            "value": converted2_sub0[0],
                        },
                        {
                            "stat": substat_translate(echo2_sub1, converted2_sub1[1]),
                            "value": converted2_sub1[0],
                        },
                        {
                            "stat": substat_translate(echo2_sub2, converted2_sub2[1]),
                            "value": converted2_sub2[0],
                        },
                        {
                            "stat": substat_translate(echo2_sub3, converted2_sub3[1]),
                            "value": converted2_sub3[0],
                        },
                        {
                            "stat": substat_translate(echo2_sub4, converted2_sub4[1]),
                            "value": converted2_sub4[0],
                        },
                    ],
                },
                {
                    "setId": "001",
                    "stat": mainstat_translate(echo3_main),
                    "statList": [
                        {
                            "stat": substat_translate(echo3_sub0, converted3_sub0[1]),
                            "value": converted3_sub0[0],
                        },
                        {
                            "stat": substat_translate(echo3_sub1, converted3_sub1[1]),
                            "value": converted3_sub1[0],
                        },
                        {
                            "stat": substat_translate(echo3_sub2, converted3_sub2[1]),
                            "value": converted3_sub2[0],
                        },
                        {
                            "stat": substat_translate(echo3_sub3, converted3_sub3[1]),
                            "value": converted3_sub3[0],
                        },
                        {
                            "stat": substat_translate(echo3_sub4, converted3_sub4[1]),
                            "value": converted3_sub4[0],
                        },
                    ],
                },
                {
                    "setId": "001",
                    "stat": mainstat_translate(echo4_main),
                    "statList": [
                        {
                            "stat": substat_translate(echo4_sub0, converted4_sub0[1]),
                            "value": converted4_sub0[0],
                        },
                        {
                            "stat": substat_translate(echo4_sub1, converted4_sub1[1]),
                            "value": converted4_sub1[0],
                        },
                        {
                            "stat": substat_translate(echo4_sub2, converted4_sub2[1]),
                            "value": converted4_sub2[0],
                        },
                        {
                            "stat": substat_translate(echo4_sub3, converted4_sub3[1]),
                            "value": converted4_sub3[0],
                        },
                        {
                            "stat": substat_translate(echo4_sub4, converted4_sub4[1]),
                            "value": converted4_sub4[0],
                        },
                    ],
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
                    "statList": [
                        {
                            "stat": echo0_sub0,
                            "value": echo0_val0,
                        },
                        {
                            "stat": echo0_sub1,
                            "value": echo0_val1,
                        },
                        {
                            "stat": echo0_sub2,
                            "value": echo0_val2,
                        },
                        {
                            "stat": echo0_sub3,
                            "value": echo0_val3,
                        },
                        {
                            "stat": echo0_sub4,
                            "value": echo0_val4,
                        },
                    ],
                },
                {
                    "setId": "001",
                    "stat": echo1_main,
                    "statList": [
                        {
                            "stat": echo1_sub0,
                            "value": echo1_val0,
                        },
                        {
                            "stat": echo1_sub1,
                            "value": echo1_val1,
                        },
                        {
                            "stat": echo1_sub2,
                            "value": echo1_val2,
                        },
                        {
                            "stat": echo1_sub3,
                            "value": echo1_val3,
                        },
                        {
                            "stat": echo1_sub4,
                            "value": echo1_val4,
                        },
                    ],
                },
                {
                    "setId": "001",
                    "stat": echo2_main,
                    "statList": [
                        {
                            "stat": echo2_sub0,
                            "value": echo2_val0,
                        },
                        {
                            "stat": echo2_sub1,
                            "value": echo2_val1,
                        },
                        {
                            "stat": echo2_sub2,
                            "value": echo2_val2,
                        },
                        {
                            "stat": echo2_sub3,
                            "value": echo2_val3,
                        },
                        {
                            "stat": echo2_sub4,
                            "value": echo2_val4,
                        },
                    ],
                },
                {
                    "setId": "001",
                    "stat": echo3_main,
                    "statList": [
                        {
                            "stat": echo3_sub0,
                            "value": echo3_val0,
                        },
                        {
                            "stat": echo3_sub1,
                            "value": echo3_val1,
                        },
                        {
                            "stat": echo3_sub2,
                            "value": echo3_val2,
                        },
                        {
                            "stat": echo3_sub3,
                            "value": echo3_val3,
                        },
                        {
                            "stat": echo3_sub4,
                            "value": echo3_val4,
                        },
                    ],
                },
                {
                    "setId": "001",
                    "stat": echo4_main,
                    "statList": [
                        {
                            "stat": echo4_sub0,
                            "value": echo4_val0,
                        },
                        {
                            "stat": echo4_sub1,
                            "value": echo4_val1,
                        },
                        {
                            "stat": echo4_sub2,
                            "value": echo4_val2,
                        },
                        {
                            "stat": echo4_sub3,
                            "value": echo4_val3,
                        },
                        {
                            "stat": echo4_sub4,
                            "value": echo4_val4,
                        },
                    ],
                },
            ],
        },
    })
