from difflib import get_close_matches
from PIL import Image
from pytesseract import image_to_string
import cv2
import numpy as np

from app.config import (
    TESSERACT_CONFIG,
    FUZZY_MATCH_CUTOFF,
    EXPECTED_IMAGE_SIZE,
    TEMPLATE_MATCH_THRESHOLD,
    NAME_LV_PATH,
)
from app.data.crops import CROPS
from app.data.avatar_names import AVATAR_NAMES
from app.data.weapon_names import WEAPON_NAMES
from app.data.mainstats import MAINSTATS
from app.data.substats import SUBSTATS

# Load template image once at import time
_name_lv_template = cv2.imread(str(NAME_LV_PATH), cv2.IMREAD_COLOR)


def _fuzzy_lookup(text: str, lookup: dict, cutoff: float = FUZZY_MATCH_CUTOFF):
    """Return the value from lookup for the best-matching key, or None."""
    cleaned = text.strip()
    if cleaned in lookup:
        return lookup[cleaned]
    matches = get_close_matches(cleaned, lookup.keys(), n=1, cutoff=cutoff)
    if matches:
        return lookup[matches[0]]
    return None


def avatar_name_to_id(text: str) -> str | None:
    return _fuzzy_lookup(text, AVATAR_NAMES)


def weapon_name_to_id(text: str) -> int | None:
    return _fuzzy_lookup(text, WEAPON_NAMES)


def mainstat_translate(text: str) -> str | None:
    return _fuzzy_lookup(text, MAINSTATS)


def substat_translate(text: str, has_percent: bool) -> str | None:
    cleaned = text.strip()

    # Try exact match first, then fuzzy
    matched_key = None
    if cleaned in SUBSTATS:
        matched_key = cleaned
    else:
        matches = get_close_matches(cleaned, SUBSTATS.keys(), n=1, cutoff=FUZZY_MATCH_CUTOFF)
        if matches:
            matched_key = matches[0]

    if matched_key is None:
        return None

    # HP/ATK/DEF are flat or percent depending on has_percent
    if matched_key in ("HP", "ATK", "DEF"):
        return f"PERCENT_{matched_key}" if has_percent else f"FLAT_{matched_key}"
    return f"PERCENT_{SUBSTATS[matched_key]}"


def value_translate(text: str) -> tuple[int | float | None, bool]:
    cleaned = text.strip()
    has_percent = cleaned.endswith("%")
    if has_percent:
        cleaned = cleaned[:-1].strip()
    try:
        value = float(cleaned)
        return (int(value) if value.is_integer() else value), has_percent
    except ValueError:
        return None, has_percent


def _ocr_crop(image: Image.Image, crop_key: str) -> str:
    """Crop the image and run OCR on the region."""
    return image_to_string(image.crop(CROPS[crop_key]), config=TESSERACT_CONFIG)


def _extract_echo(image: Image.Image, echo_index: int) -> dict:
    """Extract one echo's main stat and 5 substats."""
    prefix = f"echo{echo_index}"

    main_stat = _ocr_crop(image, f"{prefix}_main")

    substats = []
    for sub_index in range(5):
        sub_text = _ocr_crop(image, f"{prefix}_sub{sub_index}")
        val_text = _ocr_crop(image, f"{prefix}_val{sub_index}")
        value, has_percent = value_translate(val_text)
        substats.append({
            "subStatId": substat_translate(sub_text, has_percent),
            "subStatValue": value,
        })

    return {
        "mainStatId": mainstat_translate(main_stat),
        "subStatList": substats,
    }


def process_image(image: Image.Image) -> dict | None:
    """
    Run full OCR extraction on a 1920x1080 screenshot.
    Returns the structured result dict, or None if the template match fails.
    """
    if image.size != EXPECTED_IMAGE_SIZE:
        return {"error": f"Invalid image dimensions. Expected {EXPECTED_IMAGE_SIZE[0]}x{EXPECTED_IMAGE_SIZE[1]}"}

    # Template match to find the end of the avatar name
    avatar_region = np.array(image.crop(CROPS["avatar_name"]))
    avatar_bgr = cv2.cvtColor(avatar_region, cv2.COLOR_RGB2BGR)
    result = cv2.matchTemplate(avatar_bgr, _name_lv_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < TEMPLATE_MATCH_THRESHOLD:
        return None

    # Crop avatar name up to where the "LV" template was found
    avatar_name = image_to_string(
        image.crop((71, 23, 65 + max_loc[0], 89)),
        config=TESSERACT_CONFIG,
    )
    weapon_name = _ocr_crop(image, "weapon_name")

    echoes = [_extract_echo(image, i) for i in range(5)]

    return {
        "weaponId": weapon_name_to_id(weapon_name),
        "equipList": echoes,
    }
