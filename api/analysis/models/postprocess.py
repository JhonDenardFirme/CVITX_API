from io import BytesIO
from typing import Dict, Tuple, List
from PIL import Image, ImageDraw, ImageFont

def annotate(image_bytes: bytes, dets: Dict) -> bytes:
    im = Image.open(BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(im)
    boxes: List[Tuple[int,int,int,int]] = dets.get("boxes") or []
    label = f"{dets.get('make','?')} {dets.get('model','?')} ({dets.get('type','?')})"
    for (x1,y1,x2,y2) in boxes:
        draw.rectangle([x1,y1,x2,y2], outline="red", width=3)
        draw.text((x1+4, max(0,y1-14)), label, fill="red")
    out = BytesIO()
    im.save(out, format="JPEG", quality=90)
    return out.getvalue()
