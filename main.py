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
import cv2
import numpy as np
import os

nameLV = cv2.imread('nameLV.webp', cv2.IMREAD_COLOR)

def circular_mask(size=48):
    y, x = np.ogrid[:size, :size]
    center = (size - 1) / 2
    dist_from_center = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    mask = dist_from_center <= center
    return mask.astype(np.uint8)

def apply_mask_color(img, mask):
    masked = np.zeros_like(img)
    for c in range(3):
        masked[:, :, c] = cv2.bitwise_and(img[:, :, c], img[:, :, c], mask=mask)
    return masked

def load_templates(folder_path, size=48):
    templates = {}
    mask = circular_mask(size)
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".webp"):
            key = filename.split(".")[0]
            img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (size, size))
            img = apply_mask_color(img, mask)
            templates[key] = img
    return templates

def match_icon(cropped_icon, templates, mask):
    cropped_icon = apply_mask_color(cropped_icon, mask)
    best_key = None
    best_score = -1

    for key, template in templates.items():
        total_score = 0
        for c in range(3):
            result = cv2.matchTemplate(cropped_icon[:, :, c], template[:, :, c], cv2.TM_CCOEFF_NORMED)
            total_score += result[0][0]
        avg_score = total_score / 3
        if avg_score > best_score:
            best_score = avg_score
            best_key = key
    
    if best_score < 0.8:
        return None

    return best_key

mask = circular_mask()
templates = load_templates("icons", size=48)

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

def avatar_name_to_id(text: str, color: tuple) -> str | None:
    cleaned_text = text.strip()
    
    # Exact matching
    if cleaned_text in AVATAR_NAMES:
        if cleaned_text == "Rover":
            if is_color_in_range(color, (109, 143, 137)): # Aero
                return "1408"
            if is_color_in_range(color, (135, 48, 109)): # Havoc
                return "1604"
            if is_color_in_range(color, (129, 112, 79)): # Spectro
                return "1502"
            return None
        return AVATAR_NAMES[cleaned_text]
    
    # Fuzzy matching
    matches = get_close_matches(cleaned_text, AVATAR_NAMES.keys(), n=1, cutoff=0.8)
    if matches:
        if matches[0] == "Rover":
            if is_color_in_range(color, (109, 143, 137)): # Aero
                return "1408"
            if is_color_in_range(color, (135, 48, 109)): # Havoc
                return "1604"
            if is_color_in_range(color, (129, 112, 79)): # Spectro
                return "1502"
            return None
        return AVATAR_NAMES[matches[0]]
    
    return None

def weapon_name_to_id(text: str) -> str | None:
    cleaned_text = text.strip()
    
    # Exact matching
    if cleaned_text in WEAPON_NAMES:
        return WEAPON_NAMES[cleaned_text]
    
    # Fuzzy matching
    matches = get_close_matches(cleaned_text, WEAPON_NAMES.keys(), n=1, cutoff=0.8)
    if matches:
        return WEAPON_NAMES[matches[0]]
    
    return None

def mainstat_translate(text: str) -> str | None:
    cleaned_text = text.strip()
    
    # Exact matching
    if cleaned_text in MAINSTATS:
        return MAINSTATS[cleaned_text]
    
    # Fuzzy matching
    matches = get_close_matches(cleaned_text, MAINSTATS.keys(), n=1, cutoff=0.8)
    if matches:
        return MAINSTATS[matches[0]]
    
    return None

def substat_translate(text: str, has_percent: bool) -> str | None:
    cleaned_text = text.strip()
    
    # Exact matching
    if cleaned_text in SUBSTATS:
        if cleaned_text in ("HP", "ATK", "DEF"):
            if has_percent:
                return cleaned_text
            else:
                return f"_{cleaned_text}"
        return SUBSTATS[cleaned_text]
    
    # Fuzzy matching
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
        value = float(cleaned_text)
        if value.is_integer():
            return int(value), has_percent
        return value, has_percent
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
    result = cv2.matchTemplate(cv2.cvtColor(np.array(image.crop(CROPS["avatar_name"])), cv2.COLOR_RGB2BGR), nameLV, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if max_val < 0.8:
        return None
    top_left = max_loc
    h, w, _ = nameLV.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)
    avatar_name = image_to_string(image.crop((71, 23, 65 + top_left[0], 89)), config=custom_config)
    colorMain = get_average_color(image.crop(CROPS["avatar_color"]))

    avatar_color = get_average_color(image.crop(CROPS["avatar_color"]))
    weapon_name = image_to_string(image.crop(CROPS["weapon_name"]), config=custom_config)
    echo0_setIcon = cv2.cvtColor(np.array(image.crop(CROPS["echo0_set"])), cv2.COLOR_RGB2BGR)
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

    echo1_setIcon = cv2.cvtColor(np.array(image.crop(CROPS["echo1_set"])), cv2.COLOR_RGB2BGR)
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
    
    echo2_setIcon = cv2.cvtColor(np.array(image.crop(CROPS["echo2_set"])), cv2.COLOR_RGB2BGR)
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

    echo3_setIcon = cv2.cvtColor(np.array(image.crop(CROPS["echo3_set"])), cv2.COLOR_RGB2BGR)
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

    echo4_setIcon = cv2.cvtColor(np.array(image.crop(CROPS["echo4_set"])), cv2.COLOR_RGB2BGR)
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
        "id": avatar_name_to_id(avatar_name, avatar_color),
        "data": {
            "isStar": False,
            "level": 90,
            "rank": 0,
            "weaponId": weapon_name_to_id(weapon_name),
            "weaponLevel": 90,
            "weaponRank": 1,
            "skillMap": {
                "001": 10,
                "002": 10,
                "003": 10,
                "004": 10,
                "005": 10,
                "101": 1,
                "102": 1,
                "201": 1,
                "202": 1,
                "203": 1,
                "204": 1,
                "205": 1,
                "206": 1,
                "207": 1,
                "208": 1,
            },
            "equipList": [
                {
                    "setId": match_icon(echo0_setIcon, templates, mask),
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
                    "setId": match_icon(echo1_setIcon, templates, mask),
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
                    "setId": match_icon(echo2_setIcon, templates, mask),
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
                    "setId": match_icon(echo3_setIcon, templates, mask),
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
                    "setId": match_icon(echo4_setIcon, templates, mask),
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
    })
