from pathlib import Path

from rembg import remove
from PIL import Image, ImageFilter

INPUT_DIR = Path("images")
OUTPUT_DIR = Path("images_nobg")
OUTPUT_DIR.mkdir(exist_ok=True)

# Strenger = weniger Halo / weniger Randreste
ALPHA_THRESHOLD = 50

# Leichtes Abschneiden gegen schwarze Ränder
PADDING = 4


def trim_transparency(img: Image.Image, threshold: int = 50, padding: int = 4) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    alpha = img.getchannel("A")

    # Erodieren der Maske: entfernt dünne dunkle Säume
    alpha = alpha.filter(ImageFilter.MinFilter(3))

    # Schwache Randpixel entfernen
    alpha = alpha.point(lambda p: 255 if p > threshold else 0)

    bbox = alpha.getbbox()
    if bbox is None:
        return img

    left = max(bbox[0] - padding, 0)
    upper = max(bbox[1] - padding, 0)
    right = min(bbox[2] + padding, img.width)
    lower = min(bbox[3] + padding, img.height)

    return img.crop((left, upper, right, lower))


def main():
    files = sorted(INPUT_DIR.iterdir())
    count = 0

    for i, path in enumerate(files, start=1):
        if path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue

        try:
            img = Image.open(path).convert("RGBA")
            result = remove(img)
            result = trim_transparency(result, threshold=ALPHA_THRESHOLD, padding=PADDING)

            out_path = OUTPUT_DIR / f"{path.stem}.png"
            result.save(out_path)

            count += 1
            print(f"[{count}] OK: {path.name} -> {out_path.name}")

        except Exception as e:
            print(f"[{i}] ERROR: {path.name} -> {e}")

    print(f"\nDone. Saved {count} files to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
    