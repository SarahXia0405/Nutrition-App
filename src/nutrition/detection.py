from typing import List, Dict
from PIL import Image

def stub_detect(image: Image.Image) -> List[Dict]:
    """Stub for detection. Replace with YOLO/CLIP/BLIP to auto-detect foods and estimate portions.
    Return a list like:
    [
      {"food_name": "broccoli", "portion_g": 120, "confidence": 0.90},
      {"food_name": "chicken breast", "portion_g": 160, "confidence": 0.88}
    ]
    """
    return []