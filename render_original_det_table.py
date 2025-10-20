import os
from PIL import Image, ImageDraw, ImageFont

ORIG_DET = {
    "YOLOv5s": {
        "all": (0.8106, 0.3728),
        "AR": (0.8262, 0.3680),
        "MR": (0.8674, 0.4100),
        "PR": (0.7222, 0.3648),
        "TR": (0.8264, 0.3492),
    },
    "YOLOv5m": {
        "all": (0.7916, 0.3588),
        "AR": (0.8146, 0.3476),
        "MR": (0.8404, 0.3890),
        "PR": (0.6922, 0.3514),
        "TR": (0.8180, 0.3472),
    },
    "YOLOv5l": {
        "all": (0.7914, 0.3646),
        "AR": (0.8098, 0.3600),
        "MR": (0.8528, 0.4016),
        "PR": (0.6902, 0.3438),
        "TR": (0.8126, 0.3524),
    },
}


def load_font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_table(img: Image.Image, top_left=(60, 60)) -> Image.Image:
    draw = ImageDraw.Draw(img)

    title_font = load_font(28)
    header_font = load_font(20)
    body_font = load_font(18)

    x0, y0 = top_left

    # Title
    title = "Original YOLOv5 Detection (per class, averaged)"
    draw.text((x0, y0), title, fill="black", font=title_font)
    y = y0 + 40

    # Column headers
    col_models_w = 160
    col_class_w = 90
    col_map50_w = 140
    col_map5095_w = 160

    headers = [
        ("Models", col_models_w),
        ("Class", col_class_w),
        ("Original", col_map50_w + col_map5095_w),
    ]

    # Header row top
    draw.text((x0, y), headers[0][0], fill="black", font=header_font)
    draw.text((x0 + col_models_w, y), headers[1][0], fill="black", font=header_font)
    draw.text((x0 + col_models_w + col_class_w, y), "mAP@.5    mAP@.5:.95", fill="black", font=header_font)
    y += 32

    # Horizontal line
    draw.line([(x0, y), (x0 + col_models_w + col_class_w + col_map50_w + col_map5095_w, y)], fill="black", width=2)
    y += 16

    # Render rows
    def render_model_block(model_name: str, rows: list, start_y: int) -> int:
        nonlocal x0
        y_local = start_y
        # Model name (block label)
        draw.text((x0, y_local), model_name, fill="black", font=header_font)
        y_local += 6
        # For each class
        for i, (cls, (map50, map5095)) in enumerate(rows):
            # class text
            draw.text((x0 + col_models_w, y_local), f"{cls}", fill="black", font=body_font)
            # metrics
            draw.text((x0 + col_models_w + col_class_w, y_local), f"{map50:8.4f}    {map5095:8.4f}", fill="black", font=body_font)
            y_local += 26
        # separator line between models
        y_local += 10
        draw.line([(x0, y_local), (x0 + col_models_w + col_class_w + col_map50_w + col_map5095_w, y_local)], fill="#666", width=2)
        y_local += 14
        return y_local

    for model_name in ["YOLOv5s", "YOLOv5m", "YOLOv5l"]:
        items = list(ORIG_DET[model_name].items())
        y = render_model_block(model_name, items, y)

    return img


def main():
    base_path = os.getcwd()
    base_img_path = os.path.join(base_path, "files", "1760423080004@2x.jpg")
    out_img_path = os.path.join(base_path, "files", "1760423080004_origdet@2x.jpg")

    if not os.path.exists(base_img_path):
        # create blank canvas if not found
        img = Image.new("RGB", (1920, 1080), "white")
    else:
        img = Image.open(base_img_path).convert("RGBA")

    # Light overlay to improve text readability
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 210))
    img = Image.alpha_composite(img, overlay)
    img = img.convert("RGB")

    img = draw_table(img, top_left=(60, 60))

    os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
    img.save(out_img_path, quality=95)
    print(f"Saved: {out_img_path}")


if __name__ == "__main__":
    main()

