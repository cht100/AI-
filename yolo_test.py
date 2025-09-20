import glob

from ultralytics import YOLO
from PIL import Image
import cv2
import os
from pathlib import Path
import numpy as np
from ultralytics.engine.results import Boxes

def filter_overlapping_boxes(boxes, threshold=60):
    """
    根据中心点距离过滤重叠的检测框
    保留置信度高的检测框，去除中心点距离过近且置信度低的框

    Args:
        boxes: YOLO的Boxes对象
        threshold: 中心点距离阈值（像素）

    Returns:
        过滤后的Boxes对象
    """
    if boxes is None or len(boxes) <= 1:
        return boxes

    # 获取所有框的中心点和置信度
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()

    # 计算中心点坐标
    centers_x = (xyxy[:, 0] + xyxy[:, 2]) / 2
    centers_y = (xyxy[:, 1] + xyxy[:, 3]) / 2
    centers = np.column_stack((centers_x, centers_y))

    # 按置信度排序（从高到低）
    sorted_indices = np.argsort(conf)[::-1]

    keep_indices = []
    suppressed = set()

    for i in sorted_indices:
        if i in suppressed:
            continue

        keep_indices.append(i)

        # 检查与其他框的中心点距离
        current_center = centers[i]
        distances = np.sqrt(np.sum((centers - current_center) ** 2, axis=1))

        # 抑制距离过近且置信度较低的框
        for j in sorted_indices:
            if i != j and j not in suppressed and distances[j] < threshold:
                suppressed.add(j)

    # 返回过滤后的boxes
    if len(keep_indices) > 0:
        # 根据索引筛选boxes
        filtered_data = boxes.data[keep_indices]
        # 创建新的Boxes对象（这需要使用内部方法）

        return Boxes(filtered_data, boxes.orig_shape)
    else:
        return boxes


if __name__ == '__main__':
    print("开始加载模型...")
    # 加载训练好的模型
    model = YOLO('runs/detect/yolo_road_damage/weights/best.pt')

    # 查看模型信息
    print("模型架构信息:")
    model.info()

    # 查看类别信息
    print("类别名称:", model.names)
    print("类别数量:", model.model.nc)

    # 设置输入图像文件夹路径
    input_folder = 'china_d_m/images/test/'  # 输入文件夹
    output_folder = 'test_output/'  # 输出文件夹

    image_output_folder = output_folder + 'images/'
    label_output_folder = output_folder + 'labels/'

    # 创建输出文件夹
    os.makedirs(image_output_folder, exist_ok=True)
    os.makedirs(label_output_folder, exist_ok=True)

    # 获取文件夹中所有文件并只取前10个
    all_files = os.listdir(input_folder)
    image_files = all_files[:10]  # 只取前10个文件

    # 构建完整的文件路径列表
    full_paths = [os.path.join(input_folder, filename) for filename in image_files]

    # 推理 - 处理整个文件夹
    results = model.predict(source=full_paths, conf=0.1, save=False)  # 不使用自动保存

    # 遍历结果
    for i, r in enumerate(results):
        # 获取原始图像路径
        print("========================")
        print(r)
        print("========================")
        original_path = r.path  # YOLO会保存原始图像路径
        original_name = Path(original_path).name  # 获取文件名
        name_without_ext = Path(original_path).stem  # 获取不带扩展名的文件名

        # 生成保存路径
        # output_path_cv2 = os.path.join(output_folder, f"{name_without_ext}_detected.jpg")
        output_path_pil = os.path.join(image_output_folder, f"{name_without_ext}_detected_pil.jpg")

        # 绘制检测结果（只显示前5个）
        if r.boxes is not None and len(r.boxes) > 0:
            # 只取前5个检测框
            r.boxes = r.boxes[:10]  # 限制为前5个
            r.boxes = filter_overlapping_boxes(r.boxes)
            # 创建一个新的Results对象或手动绘制
            im_array = r.plot()  # 这会绘制所有框，我们需要自定义绘制
        else:
            im_array = r.plot()  # 没有检测框时直接绘制

        """
        # 使用OpenCV保存（需要RGB转BGR）
        im_bgr = im_array[..., ::-1]  # RGB to BGR
        cv2.imwrite(output_path_cv2, im_bgr)
        print(f"✅ OpenCV保存图像: {output_path_cv2}")
        """

        # 使用PIL保存
        im_pil = Image.fromarray(im_array)
        im_pil.save(output_path_pil)
        print(f"✅ PIL保存图像: {output_path_pil}")

        # 保存检测结果为文本文件
        txt_path = os.path.join(label_output_folder, f"{name_without_ext}_detected.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"检测到的目标 (图像: {original_name}):\n")
            for j, box in enumerate(r.boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                f.write(f"目标{j + 1}: 类别={model.names[cls]}, 置信度={conf:.2f}, 边界框={xyxy}\n")
        print(f"✅ 保存检测结果: {txt_path}")

        # 打印检测到的目标
        print(f"图像 {original_name} 检测到的目标：")
        for j, box in enumerate(r.boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            print(f"  目标{j + 1}: 类别={model.names[cls]}, 置信度={conf:.2f}, 边界框={xyxy}")
        print("-" * 50)
