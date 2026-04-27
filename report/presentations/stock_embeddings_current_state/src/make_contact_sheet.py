from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "scratch"


def main() -> None:
    files = sorted(SCRATCH.glob("pptx_slide_*.png"))
    thumb_w, thumb_h = 480, 270
    margin = 28
    label_h = 28
    cols = 3
    rows = (len(files) + cols - 1) // cols
    sheet_w = cols * (thumb_w + margin) + margin
    sheet_h = rows * (thumb_h + label_h + margin) + margin

    sheet = Image.new("RGB", (sheet_w, sheet_h), (244, 236, 217))
    draw = ImageDraw.Draw(sheet)

    for idx, file_path in enumerate(files):
        image = Image.open(file_path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h))
        thumb = Image.new("RGB", (thumb_w, thumb_h), (255, 248, 232))
        thumb.paste(image, ((thumb_w - image.width) // 2, (thumb_h - image.height) // 2))

        x = margin + (idx % cols) * (thumb_w + margin)
        y = margin + (idx // cols) * (thumb_h + label_h + margin)
        sheet.paste(thumb, (x, y))
        draw.text(
            (x, y + thumb_h + 6),
            file_path.stem.replace("pptx_slide_", "Slide "),
            fill=(23, 32, 27),
        )

    sheet.save(SCRATCH / "pptx_contact_sheet.png")


if __name__ == "__main__":
    main()
