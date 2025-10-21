import os
import json
import sys
import gradio as gr
import shutil

from PIL import Image, ImageDraw
from loguru import logger
from pathlib import Path

"""
xCensorNing 打码工具 - 增强版
支持：
✅ 椭圆扁度调节
✅ 垂直偏移调节
✅ 遮罩范围放大倍率
✅ 马赛克颗粒粗度可控
"""

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
def file_path2name(path) -> str:
    return os.path.basename(path)

def file_path2list(path) -> list[str]:
    return os.listdir(path)

def file_namel2pathl(file_list: list, file_path):
    return [Path(file_path) / file for file in file_list]


NEIGHBOR = 0.0025

# -------------------- #
# 检测模型加载
# -------------------- #
try:
    from ultralytics import YOLO
    logger.debug("使用 YOLO 进行图像预测")

    def detector(image):
        model = YOLO("./models/censor.pt")
        box_list = []
        results = model(image, verbose=False)
        result = json.loads(results[0].to_json())
        for part in result:
            if part["name"] in ["penis", "pussy"]:
                logger.debug(f"检测到: {part['name']}")
                x = round(part["box"]["x1"])
                y = round(part["box"]["y1"])
                w = round(part["box"]["x2"] - part["box"]["x1"])
                h = round(part["box"]["y2"] - part["box"]["y1"])
                box_list.append([x, y, w, h])
        return box_list

except ModuleNotFoundError:
    from nudenet import NudeDetector
    logger.debug("使用 NudeNet 进行图像检测")

    def detector(image):
        nude_detector = NudeDetector()
        box_list = []
        body = nude_detector.detect("./output/temp.png")
        for part in body:
            if part["class"] in ["FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED"]:
                logger.debug(f"检测到: {part['class']}")
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
    length = img.width if img.width > img.height else img.height
    c = img.crop((fx, fy, tx, ty))
    c = __mosaic_blurry(c, length)
    img.paste(c, (fx, fy, tx, ty))
    return img

def mosaic_blurry(img):
    img = str(img)
    with Image.open(img) as image:
        box_list = detector(img)
        for box in box_list:
            image = _mosaic_blurry(image, box[0], box[1], box[0] + box[2], box[1] + box[3])
            image.save(img)


# -------------------- #
# 像素化椭圆模式（可调扁度、偏移与范围）
# -------------------- #
def _mosaic_pixel_ellipse(image, box, block_size,
                          scale=0.8, aspect=0.6,
                          offset_y_ratio=-0.1,
                          scale_multiplier=1.2):
    """
    对检测到的区域应用椭圆像素化马赛克。
    - block_size: 马赛克颗粒大小
    - scale: 椭圆基础缩放比例
    - aspect: 椭圆高度压缩比例（越小越扁）
    - offset_y_ratio: 垂直偏移比例（负值上移）
    - scale_multiplier: 遮罩范围倍率（越大覆盖越多）
    """
    x, y, w, h = box
    region = (x, y, x + w, y + h)

    cropped = image.crop(region)
    small = cropped.resize(
        (max(1, int(w / block_size)), max(1, int(h / block_size))),
        resample=Image.Resampling.NEAREST
    )
    mosaic_img = small.resize(cropped.size, Image.Resampling.NEAREST)

    mask = Image.new("L", cropped.size, 0)
    draw = ImageDraw.Draw(mask)

    ellipse_w = int(w * scale * scale_multiplier)
    ellipse_h = int(h * scale * aspect * scale_multiplier)
    offset_x = (w - ellipse_w) // 2
    offset_y = int((h - ellipse_h) // 2 + h * offset_y_ratio)

    draw.ellipse(
        (offset_x, offset_y, offset_x + ellipse_w, offset_y + ellipse_h),
        fill=255
    )

    image.paste(mosaic_img, region, mask)
    return image


def mosaic_pixel(img_path, aspect=0.6, offset_y_ratio=-0.1, scale_multiplier=1.2):
    img_path = str(img_path)
    box_list = detector(img_path)

    for box in box_list:
        with Image.open(img_path) as pil_img:
            neighbor = int(
                pil_img.width * NEIGHBOR if pil_img.width > pil_img.height else pil_img.height * NEIGHBOR
            )
            image = _mosaic_pixel_ellipse(
                pil_img, box, neighbor,
                scale=0.8,
                aspect=aspect,
                offset_y_ratio=offset_y_ratio,
                scale_multiplier=scale_multiplier
            )
            image.save(img_path)


# -------------------- #
# 线条模式
# -------------------- #
def mosaic_lines(img_path):
    img_path = str(img_path)
    box_list = detector(img_path)
    with Image.open(img_path) as image:
        draw = ImageDraw.Draw(image)
        for box in box_list:
            x, y, w, h = box
            while y <= box[1] + box[3]:
                xy = [(x, y), (x + w, y)]
                draw.line(xy, fill="black", width=int(10 * 0.35))
                y += int(box[3] * 0.15)
        image.save(img_path)


# -------------------- #
# 主处理函数
# -------------------- #
def process_images_gradio(input_folder_path, mosaic_type,
                          neighbor_value_ui, aspect_value_ui,
                          offset_value_ui, scale_multiplier_ui):
    global NEIGHBOR
    NEIGHBOR = float(neighbor_value_ui)
    aspect_value_ui = float(aspect_value_ui)
    offset_value_ui = float(offset_value_ui)
    scale_multiplier_ui = float(scale_multiplier_ui)
    logger.info(f"参数：NEIGHBOR={NEIGHBOR}, aspect={aspect_value_ui}, offset={offset_value_ui}, scale_mul={scale_multiplier_ui}")

    if not input_folder_path:
        return "❌ 错误：请输入文件夹路径。"

    input_path = Path(input_folder_path)
    if not input_path.is_dir():
        return f"❌ 错误：'{input_folder_path}' 不是有效目录。"

    output_folder = Path("./output")
    output_folder.mkdir(parents=True, exist_ok=True)

    processed_files_count = 0
    error_messages = []

    try:
        image_files = [f for f in os.listdir(input_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))]

        if not image_files:
            return "ℹ️ 指定文件夹中没有找到图片文件。"

        for filename in image_files:
            original_img_path = input_path / filename
            output_img_path = output_folder / filename

            try:
                shutil.copy2(original_img_path, output_img_path)
                logger.info(f"处理 {filename} 使用模式: {mosaic_type}")

                if mosaic_type == "模糊 (Blurry)":
                    mosaic_blurry(str(output_img_path))
                elif mosaic_type == "像素化 (Pixelated)":
                    mosaic_pixel(str(output_img_path),
                                 aspect=aspect_value_ui,
                                 offset_y_ratio=offset_value_ui,
                                 scale_multiplier=scale_multiplier_ui)
                elif mosaic_type == "线条 (Lines)":
                    mosaic_lines(str(output_img_path))

                processed_files_count += 1
            except Exception as e:
                logger.error(f"处理文件 '{filename}' 时出错: {e}")
                error_messages.append(f"{filename}: {str(e)}")

        status_message = f"处理完成！成功处理 {processed_files_count}/{len(image_files)} 张图片。\n输出目录: {output_folder.resolve()}"
        if error_messages:
            status_message += "\n错误详情:\n" + "\n".join(error_messages)
        return status_message

    except Exception as e:
        logger.error(f"处理过程中发生严重错误: {e}")
        return f"处理过程中发生严重错误: {str(e)}"


# -------------------- #
# 启动 Gradio 界面
# -------------------- #
if __name__ == "__main__":
    iface = gr.Interface(
        fn=process_images_gradio,
        inputs=[
            gr.Textbox(label="📂 输入图片文件夹路径", placeholder="例如: C:\\Users\\YourName\\Pictures"),
            gr.Radio(choices=["模糊 (Blurry)", "像素化 (Pixelated)", "线条 (Lines)"],
                     label="选择打码模式", value="像素化 (Pixelated)"),
            gr.Number(label="🧩 NEIGHBOR 值 (像素化强度)", value=NEIGHBOR,
                      minimum=0.0001, maximum=0.1, step=0.0001,
                      info="值越小格子越大。推荐 0.001 - 0.05"),
            gr.Slider(label="🔘 椭圆扁度 (aspect)", minimum=0.3, maximum=1.0, step=0.05, value=0.6,
                      info="控制椭圆的扁度，越小越扁。"),
            gr.Slider(label="↕ 垂直偏移 (offset_y_ratio)", minimum=-0.3, maximum=0.3, step=0.01, value=-0.1,
                      info="控制马赛克区域上下偏移，负值上移。"),
            gr.Slider(label="📏 遮罩范围倍率 (scale_multiplier)", minimum=0.8, maximum=2.0, step=0.1, value=1.2,
                      info="控制马赛克区域整体放大。")
        ],
        outputs=gr.Textbox(label="状态输出", lines=7, interactive=False),
        title="xCensorNing 图片打码工具 (增强版)",
        description=(
            "使用说明：\n"
            "1输入图片文件夹路径\n"
            "2选择打码模式\n"
            "3调整像素化强度、椭圆扁度、垂直偏移与范围倍率\n"
            "4点击 Submit 开始处理\n\n"
            "输出结果将保存在脚本同目录的 output 文件夹中。"
        ),
        allow_flagging="never",
        theme=gr.themes.Soft(),
        live=False
    )

    logger.info("🚀 启动 Gradio 界面：http://127.0.0.1:2333")
    iface.launch(server_name="127.0.0.1", server_port=2333)
