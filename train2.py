import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('E:/zhs_cell/ultralytics-main/ultralytics-main/ultralytics-main/datasets/cell.yaml')
    model = YOLO('E:/DeepLearning/YOLO/YOLOv8/ultralytics-main/ultralytics-main/ultralytics/cfg/models/11/yolo11.yaml')

    model.train(data='E:/DeepLearning/YOLO/YOLOv8/ultralytics-main/ultralytics-main/ImageSets_4_class/my_data.yaml')