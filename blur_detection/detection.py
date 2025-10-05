import cv2
import numpy as np
from ultralytics import YOLO
import os

class BirdSharpnessAnalyzer:
    """
    一个用于分析图片中鸟类清晰度的类。
    - 使用 YOLOv8-seg 进行实例分割。
    - 计算每只鸟的拉普拉斯方差作为清晰度得分。
    """

    def __init__(self, model_path='yolov8n-seg.pt'):
        """
        初始化分析器并加载 YOLOv8 分割模型。

        Args:
            model_path (str): YOLOv8 分割模型的路径。'yolov8n-seg.pt' 是一个轻量级模型。
                               也可以使用 'yolov8s-seg.pt', 'yolov8m-seg.pt' 等更大更精确的模型。
        """
        print(f"正在加载模型: {model_path}")
        self.model = YOLO(model_path)
        # COCO 数据集中 'bird' 类别的索引是 14
        self.bird_class_id = 14
        print("模型加载完成。")

    def calculate_sharpness(self, image, mask):
        """
        在给定的掩码区域内计算图像的清晰度。

        Args:
            image (np.array): BGR 格式的原始图像。
            mask (np.array): 单通道的二值掩码，标记了鸟的区域。

        Returns:
            float: 拉普拉斯方差值，即清晰度得分。
        """
        # 将图像转换为灰度图
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 分别计算x和y方向的梯度
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

        # 使用掩码提取属于鸟的像素
        # 我们只对这部分像素计算清晰度
        grad_x_bird = grad_x[mask != 0]
        grad_y_bird = grad_y[mask != 0]

        if grad_x_bird.size == 0 or grad_y_bird.size==0:
            return 0.0


        # 计算梯度方差
        var_x = np.var(grad_x_bird)
        var_y = np.var(grad_y_bird)

        # 使用两个方差的最小值作为最终得分
        sharpness_score = min(var_x, var_y)

        return sharpness_score

    def analyze_image(self, image_path, conf_threshold=0.4):
        """
        分析单张图片，检测鸟类并计算清晰度。

        Args:
            image_path (str): 输入图片的路径。
            conf_threshold (float): 用于过滤检测结果的置信度阈值。

        Returns:
            tuple: 包含处理后图像和结果列表的元组 (visualized_image, results_data)。
        """
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"错误: 无法读取图片 {image_path}")
            return None, []

        # 运行 YOLOv8 推理
        # print("正在进行推理...")
        results = self.model(image, verbose=False)  # verbose=False 关闭冗长的输出

        # 复制一份原图用于绘制结果
        visualized_image = image.copy()

        # 存储结果
        results_data = []

        # 解析推理结果
        result = results[0]  # 获取第一张图片的结果

        # 检查是否有掩码输出
        if result.masks is None:
            # print("未检测到任何物体。")
            return visualized_image, results_data

        # 获取所有检测框、类别和掩码
        boxes = result.boxes.cpu().numpy()
        masks = result.masks.data.cpu().numpy()

        # print(f"检测到 {len(boxes)} 个物体，正在筛选鸟类...")

        # 遍历每个检测到的物体
        for i in range(len(boxes)):
            class_id = int(boxes[i].cls[0])
            confidence = boxes[i].conf[0]

            # 只处理类别为 'bird' 且置信度高于阈值的物体
            if class_id == self.bird_class_id and confidence > conf_threshold:
                # 获取边界框坐标
                x1, y1, x2, y2 = map(int, boxes[i].xyxy[0])

                # 获取并处理掩码
                mask_raw = masks[i]
                # 将掩码尺寸调整为原图大小，并转换为二值图像 (0 或 1)
                mask = cv2.resize(mask_raw, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(
                    'uint8')

                # 计算清晰度
                sharpness_score = self.calculate_sharpness(image, mask)

                # 存储结果
                results_data.append({
                    'box': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'sharpness': sharpness_score
                })

                # --- 可视化 ---
                # 绘制边界框
                cv2.rectangle(visualized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 绘制分割轮廓
                # 将掩码转换为OpenCV可用的格式
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(visualized_image, contours, -1, (255, 0, 0), 2)  # 蓝色轮廓

                # 在边界框上方显示清晰度得分
                label = f"Bird: {confidence:.2f} | Sharpness: {sharpness_score:.2f}"
                cv2.putText(visualized_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return visualized_image, results_data


if __name__ == '__main__':
    # --- 使用示例 ---

    # 1. 创建分析器实例 (模型会自动下载)
    analyzer = BirdSharpnessAnalyzer(model_path='../yolov8l-seg.pt')  # 使用稍大一点的模型以获得更高精度

    # 2. 指定要分析的图片路径
    # 请将 'your_bird_image.jpg' 替换为你的图片文件路径
    input_image_path = r'./2025-09-21/P21A2188.JPG'

    # for input_image_path in
    # 3. 运行分析
    visualized_image, analysis_results = analyzer.analyze_image(input_image_path)

    # 4. 处理结果
    if visualized_image is not None:
        print("\n--- 分析结果 ---")
        if not analysis_results:
            print("在图片中没有找到符合条件的鸟。")
        else:
            for i, result in enumerate(analysis_results):
                print(f"鸟 {i + 1}:")
                print(f"  - 置信度: {result['confidence']:.4f}")
                print(f"  - 清晰度得分: {result['sharpness']:.2f}")

        # 5. 保存并显示结果图片
        output_image_path = f'output_bird_analysis_{os.path.splitext(os.path.basename(input_image_path))[0]}.jpg'
        cv2.imwrite(output_image_path, visualized_image)
        print(f"\n结果已保存至: {output_image_path}")

        # 如果需要，可以取消下一行注释以直接显示图片
        # cv2.imshow("Bird Sharpness Analysis", visualized_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()