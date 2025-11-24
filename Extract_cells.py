from ultralytics import YOLO
import cv2
import os

# 路径配置
input_folder = 'I:/3/Cell_detection_analysis_2'
output_u87 = 'G:/cellular_mechanics/cell_extract/output/U87'
output_nha = 'G:/cellular_mechanics/cell_extract/output/NHA'
os.makedirs(output_u87, exist_ok=True)
os.makedirs(output_nha, exist_ok=True)

# 加载模型
model = YOLO('G:/cellular_mechanics/消融实验/nbam/nbam5/weights/best.pt')  # 替换为你训练好的模型路径

# 初始化编号
u87_count, nha_count = 0, 0

# 遍历所有bmp图像
for filename in os.listdir(input_folder):
    if filename.endswith('.bmp'):
        img_path = os.path.join(input_folder, filename)
        results = model(img_path)
        img = cv2.imread(img_path)

        cross_boxes = []

        # 第一次遍历：收集所有 Cross 框
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                if label == 'Cross':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cross_boxes.append((x1, y1, x2, y2))

        # 第二次遍历：处理 U87 和 NHA
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]

                if label in ['U87', 'NHA']:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # 判断中心点是否在任意一个 Cross 框内
                    inside_cross = any(xc1 <= cx <= xc2 and yc1 <= cy <= yc2 for xc1, yc1, xc2, yc2 in cross_boxes)

                    if not inside_cross:
                        crop = img[y1:y2, x1:x2]
                        if label == 'U87':
                            u87_count += 1
                            save_path = os.path.join(output_u87, f'U87_{u87_count:04d}.bmp')
                            cv2.imwrite(save_path, crop)
                        elif label == 'NHA':
                            nha_count += 1
                            save_path = os.path.join(output_nha, f'NHA_{nha_count:04d}.bmp')
                            cv2.imwrite(save_path, crop)
