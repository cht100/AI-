from ultralytics.models import YOLO
from ultralytics.nn.tasks import DetectionModel
import inspect



if __name__ == '__main__':
    # 获取模型定义文件路径
    #print("模型定义文件:", inspect.getfile(DetectionModel))

    # 加载预训练模型
    model = YOLO('yolov8n.pt')
    #print("model:",model.model)
    print("info:",model.info())

    model.train(
        data='road_damage.yaml',
        epochs=30,
        imgsz=512,
        batch=16,
        augment=True,
        patience=10,
        optimizer='Adam',
        lr0=0.001,

        # 学习率调度
        lrf=0.01,  # 最终学习率比例因子

        # Warmup设置
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # 其他优化参数
        momentum=0.937,
        weight_decay=0.0005,

        name="yolo_road_damage"
    )

    """
    
    """



