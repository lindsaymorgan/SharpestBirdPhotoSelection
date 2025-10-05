import subprocess
import json
from PIL import Image, ImageDraw
import rawpy
from datetime import datetime
import os

def draw_af_areas(image_path, af_data):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # 处理单点或多点对焦
    x_positions = af_data["AFAreaXPositions"][0] if isinstance(af_data["AFAreaXPositions"], list) else [
        af_data["AFAreaXPositions"]]
    y_positions = af_data["AFAreaYPositions"][0]  if isinstance(af_data["AFAreaYPositions"], list) else [
        af_data["AFAreaYPositions"]]
    widths = af_data["AFAreaWidths"][0]  if isinstance(af_data["AFAreaWidths"], list) else [af_data["AFAreaWidths"]]
    heights = af_data["AFAreaHeights"][0]  if isinstance(af_data["AFAreaHeights"], list) else [af_data["AFAreaHeights"]]

    for x, y, w, h in zip(x_positions, y_positions, widths, heights):
        left = x - w // 2
        top = y - h // 2
        right = x + w // 2
        bottom = y + h // 2
        draw.rectangle([left, top, right, bottom], outline="red", width=2)

    img.show()

def get_af_info(cr3_file):
    cmd = ["exiftool", "-j", cr3_file]#"-AF*",
    result = subprocess.run(cmd, capture_output=True, text=True)
    metadata = json.loads(result.stdout)[0]

    # 提取 AF 相关字段
    af_info = {k: v for k, v in metadata.items() if k.startswith("AF")}
    time=metadata["TimeStamp"]
    return datetime.strptime(time, '%Y:%m:%d %H:%M:%S.%f'),af_info


def extract_and_draw_af(output_path,cr3_file):
    # 提取 AF 数据
    time,af_data = get_af_info(cr3_file)

    # 提取 JPEG 预览图
    with rawpy.imread(cr3_file) as raw:
        rgb = raw.postprocess(use_camera_wb=True)
        preview = Image.fromarray(rgb)
        # preview.save("preview.jpg")

    # 缩放坐标（RAW 尺寸 → 预览图尺寸）
    with rawpy.imread(cr3_file) as raw:
        raw_width, raw_height = raw.sizes.width, raw.sizes.height
    preview_width, preview_height = preview.size
    scale_x = preview_width / raw_width
    scale_y = preview_height / raw_height

    photosize=af_data["AFImageWidth"], af_data["AFImageHeight"]


    # 处理多对焦点
    x_pos = [int(x) for x in str(af_data["AFAreaXPositions"]).split()]
    y_pos = [int(y) for y in str(af_data["AFAreaYPositions"]).split()]
    widths = [int(w) for w in str(af_data["AFAreaWidths"]).split()]
    heights = [int(h) for h in str(af_data["AFAreaHeights"]).split()]

    # 缩放坐标
    x = [int(x * scale_x) for x in x_pos][0]
    y = [int(y * scale_y) for y in y_pos][0]
    w = [int(w * scale_x) for w in widths][0]
    h = [int(h * scale_y) for h in heights][0]

    # 绘制矩形
    draw = ImageDraw.Draw(preview)
    # for x, y, w, h in (x_pos_scaled[0], y_pos_scaled[0], widths_scaled[0], heights_scaled[0]):
    left = photosize[0]/2+(x - w // 2)
    top = photosize[1]/2-(y + h // 2)
    right = photosize[0]/2+(x + w // 2)
    bottom = photosize[1]/2-(y - h // 2)
    draw.rectangle([left, top, right, bottom], outline="red", width=3)

    # preview.show()
    preview.save(f"{output_path}/af_marked_{os.path.basename(cr3_file).split('.')[0]}.jpg")
    return time



if __name__ == "__main__":
    cr3_file = r"D:\photos\P21A0001.CR3"
    extract_and_draw_af(r"D:/photos/",cr3_file)


    # 调用
    # extract_and_draw_af(cr3_file)
    # af_data = get_af_info(cr3_file)
    # draw_af_areas(cr3_file, af_data)
    # print("AF 信息:", json.dumps(af_data, indent=2, ensure_ascii=False))