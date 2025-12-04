from ultralytics import YOLO
from PIL import Image
from .models import Detection
import torch

class YOLOv5Detector:
    def __init__(self, weights_path='weights/best.pt', conf_thres=0.4, iou_thres=0.6):
        print(f"Loading model from {weights_path} using Ultralytics API...")
        try:
            # 核心修改：不再使用 torch.hub，而是直接用 ultralytics.YOLO 加载
            # 这会自动处理版本兼容问题 (grid error)
            self.model = YOLO(weights_path) 
            self.conf_thres = conf_thres
            self.iou_thres = iou_thres
            
            # 这里定义你要检测的类别名称
            # 如果你的 best.pt 训练时只有一类，通常不需要改
            # 如果需要过滤，可以在 detect 方法里做
            self.colors = [(0, 122, 255)]  # 默认蓝色 (RGB)
            
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def detect(self, image: Image.Image) -> list[Detection]:
        if self.model is None:
            return []

        # 执行推理
        # ultralytics 接受 PIL 图片
        # conf: 置信度阈值, iou: NMS 阈值
        results = self.model(image, conf=self.conf_thres, iou=self.iou_thres, verbose=False)
        
        detections = []
        
        # 解析结果 (Ultralytics 的结果格式)
        for result in results:
            boxes = result.boxes  # 获取所有的检测框
            for box in boxes:
                # 1. 获取坐标 (x1, y1, x2, y2)
                # xyxy 是 tensor，需要转为 list 并取整
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, coords)
                
                # 2. 获取置信度
                conf = float(box.conf[0])
                
                # 3. 获取类别名称
                cls_id = int(box.cls[0])
                label_name = result.names[cls_id] if result.names else str(cls_id)
                
                # 4. 颜色 (默认取第一个颜色)
                color = self.colors[0]

                detections.append(Detection(
                    box=[x1, y1, x2, y2],
                    confidence=conf,
                    label=label_name,
                    color=color
                ))
            
        return detections