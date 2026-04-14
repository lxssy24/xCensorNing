import os
import json
import sys
import gradio as gr
import shutil

from PIL import Image, ImageDraw
from loguru import logger
from pathlib import Path

# -------------------- #
# 日志配置
# -------------------- #
format_ = (
    f"<m>xCensorNing</m>"
    "| <c>{time:YY-MM-DD HH:mm:ss}</c> "
    "| <c>{module}:{line}</c> "
    "| <level>{level}</level> "
    "| <level>{message}</level>"
)

logger.remove()
logger.add(sys.stdout, format=format_, colorize=True)

# -------------------- #
# 工具函数
# -------------------- #
def remove_exif(img_path: str):
    with Image.open(img_path) as im:
        data = list(im.getdata())
        clean = Image.new(im.mode, im.size)
        clean.putdata(data)
        clean.save(img_path)
    logger.info(f"已清除 EXIF: {img_path}")


NEIGHBOR = 0.0025

# -------------------- #
# 检测模型（修复：只加载一次）
# -------------------- #
try:
    from ultralytics import YOLO
    logger.debug("使用 YOLO 进行图像预测")

    model = YOLO("./models/censor.pt")  # ✅ 只加载一次

    def detector(image):
        box_list = []
        results = model(image, verbose=False)
        result = json.loads(results[0].to_json())

        for part in result:
            if part["name"] in ["penis", "pussy"]:
                x = round(part["box"]["x1"])
                y = round(part["box"]["y1"])
                w = round(part["box"]["x2"] - part["box"]["x1"])
                h = round(part["box"]["y2"] - part["box"]["y1"])
                box_list.append([x, y, w, h])

        return box_list

except ModuleNotFoundError:
    from nudenet import NudeDetector
    logger.debug("使用 NudeNet 进行图像检测")

    nude_detector = NudeDetector()

    def detector(image):
        box_list = []
        body = nude_detector.detect(image)  # ✅ 修复路径问题

        for part in body:
            if part["class"] in ["FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED"]:
                x, y, w, h = part["box"]
                box_list.append([x, y, w, h])

        return box_list


# -------------------- #
# 模糊模式
# -------------------- #
def __mosaic_blurry(img, length):
    s = img.size
    img = img.resize((int(length * 0.01), int(length * 0.01)))
    img = img.resize(s)
    return img


def _mosaic_blurry(img, fx, fy, tx, ty):
    length = max(img.width, img.height)
    c = img.crop((fx, fy, tx, ty))
    c = __mosaic_blurry(c, length)
    img.paste(c, (fx, fy, tx, ty))
    return img


def mosaic_blurry(img):
    with Image.open(img) as image:
        box_list = detector(img)
        for box in box_list:
            image = _mosaic_blurry(image, box[0], box[1], box[0]+box[2], box[1]+box[3])
        image.save(img)


# -------------------- #
# 像素化椭圆模式
# -------------------- #
def _mosaic_pixel_ellipse(image, box, block_size,
                          scale=0.8, aspect=0.6,
                          offset_y_ratio=-0.1,
                          scale_multiplier=1.2):

    x, y, w, h = box
    region = (x, y, x+w, y+h)

    cropped = image.crop(region)
    small = cropped.resize(
        (max(1, int(w/block_size)), max(1, int(h/block_size))),
        resample=Image.Resampling.NEAREST
    )
    mosaic_img = small.resize(cropped.size, Image.Resampling.NEAREST)

    mask = Image.new("L", cropped.size, 0)
    draw = ImageDraw.Draw(mask)

    ellipse_w = int(w * scale * scale_multiplier)
    ellipse_h = int(h * scale * aspect * scale_multiplier)
    offset_x = (w - ellipse_w) // 2
    offset_y = int((h - ellipse_h) // 2 + h * offset_y_ratio)

    draw.ellipse((offset_x, offset_y, offset_x+ellipse_w, offset_y+ellipse_h), fill=255)

    image.paste(mosaic_img, region, mask)
    return image


def mosaic_pixel(img_path, aspect=0.6, offset_y_ratio=-0.1, scale_multiplier=1.2):
    box_list = detector(img_path)

    for box in box_list:
        with Image.open(img_path) as pil_img:
            neighbor = int(max(pil_img.width, pil_img.height) * NEIGHBOR)

            image = _mosaic_pixel_ellipse(
                pil_img, box, neighbor,
                aspect=aspect,
                offset_y_ratio=offset_y_ratio,
                scale_multiplier=scale_multiplier
            )
            image.save(img_path)


# -------------------- #
# 线条模式
# -------------------- #
def mosaic_lines(img_path):
    box_list = detector(img_path)

    with Image.open(img_path) as image:
        draw = ImageDraw.Draw(image)

        for box in box_list:
            x, y, w, h = box
            yy = y
            while yy <= y + h:
                draw.line([(x, yy), (x+w, yy)], fill="black", width=int(10*0.35))
                yy += int(h * 0.15)

        image.save(img_path)


# -------------------- #
# 主处理逻辑
# -------------------- #
def process_images_gradio(input_folder_path, mosaic_type,
                          neighbor_value_ui, aspect_value_ui,
                          offset_value_ui, scale_multiplier_ui):

    global NEIGHBOR
    NEIGHBOR = float(neighbor_value_ui)

    input_path = Path(input_folder_path)
    if not input_path.is_dir():
        return "❌ 输入路径无效"

    output_folder = Path("./output")
    output_folder.mkdir(exist_ok=True)

    count = 0

    for filename in os.listdir(input_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            continue

        src = input_path / filename
        dst = output_folder / filename

        shutil.copy2(src, dst)

        if mosaic_type == "模糊 (Blurry)":
            mosaic_blurry(str(dst))
        elif mosaic_type == "像素化 (Pixelated)":
            mosaic_pixel(str(dst), aspect_value_ui, offset_value_ui, scale_multiplier_ui)
        elif mosaic_type == "线条 (Lines)":
            mosaic_lines(str(dst))

        remove_exif(str(dst))
        count += 1

    return f"✅ 完成 {count} 张 → {output_folder.resolve()}"


# -------------------- #
# Gradio 4.x UI
# -------------------- #
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("# xCensorNing 图片打码工具")

    input_path = gr.Textbox(label="输入文件夹路径")
    mosaic_type = gr.Radio(
        ["模糊 (Blurry)", "像素化 (Pixelated)", "线条 (Lines)"],
        value="像素化 (Pixelated)"
    )

    neighbor = gr.Number(value=NEIGHBOR, label="NEIGHBOR")
    aspect = gr.Slider(0.3, 1.0, value=0.6, label="aspect")
    offset = gr.Slider(-0.3, 0.3, value=-0.1, label="offset")
    scale = gr.Slider(0.8, 2.0, value=1.2, label="scale")

    btn = gr.Button("开始处理")
    output = gr.Textbox()

    btn.click(
        process_images_gradio,
        inputs=[input_path, mosaic_type, neighbor, aspect, offset, scale],
        outputs=output
    )


if __name__ == "__main__":
    logger.info("🚀 启动：http://127.0.0.1:2333")
    demo.launch(server_name="127.0.0.1", server_port=2333)
