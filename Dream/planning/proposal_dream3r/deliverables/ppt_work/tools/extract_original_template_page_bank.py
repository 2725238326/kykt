from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageDraw
import win32com.client


BASE = Path(r"E:\kykt\Dream\planning\proposal_dream3r\deliverables\ppt_work")
SRC = Path(r"E:\Work\HSY\学术风ppt模板-蓝色.pptx")
OUT = BASE / "platform_original_template_page_bank.pptx"
PREVIEW_DIR = BASE / "previews_platform_original_template_page_bank"
CONTACT = BASE / "contact_sheet_platform_original_template_page_bank.png"

# Original slide numbers in 学术风ppt模板-蓝色.pptx.
# These are kept untouched as a page bank for manual filling.
KEEP = [
    3,   # cover
    5,   # outline
    15,  # three-column problem/status/need
    25,  # left/right figure + bullets
    30,  # four figure cards
    31,  # three-module route
    32,  # process / comparison style
    37,  # three-column cards + bottom note
    48,  # text + chart/result
    49,  # technical route / flow
    51,  # table-like framework
    64,  # work content / pictures
    67,  # phased plan
    68,  # schedule / indicators
    70,  # summary-style blocks
    72,  # thanks
]


def main() -> None:
    shutil.copy2(SRC, OUT)
    app = win32com.client.Dispatch("PowerPoint.Application")
    app.Visible = True
    pres = app.Presentations.Open(str(OUT), WithWindow=False)

    keep_set = set(KEEP)
    for i in range(pres.Slides.Count, 0, -1):
        if i not in keep_set:
            pres.Slides(i).Delete()

    pres.Save()
    if PREVIEW_DIR.exists():
        shutil.rmtree(PREVIEW_DIR)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    pres.Export(str(PREVIEW_DIR), "PNG", 1920, 1080)
    pres.Close()
    app.Quit()

    def slide_no(path: Path) -> int:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return int(digits) if digits else 0

    files = sorted(PREVIEW_DIR.glob("*.PNG"), key=slide_no)
    thumbs = []
    for idx, f in enumerate(files, start=1):
        im = Image.open(f).convert("RGB")
        im.thumbnail((360, 203), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (360, 226), "white")
        canvas.paste(im, (0, 23))
        d = ImageDraw.Draw(canvas)
        d.rectangle([0, 0, 359, 225], outline=(210, 220, 230))
        d.text((8, 4), f"Bank {idx:02d} / Source {KEEP[idx - 1]:02d}", fill=(0, 70, 130))
        thumbs.append(canvas)

    cols = 4
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 360, rows * 226), (245, 247, 250))
    for idx, im in enumerate(thumbs):
        sheet.paste(im, ((idx % cols) * 360, (idx // cols) * 226))
    sheet.save(CONTACT)

    print(OUT)
    print(PREVIEW_DIR)
    print(CONTACT)


if __name__ == "__main__":
    main()
