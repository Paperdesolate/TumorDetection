# 此文档用于整理数据集，只需更换Data路径和目标标签和文件的路径

import os
import shutil
import random
from tqdm import tqdm


def split_img(img_path, label_path, split_list):
    try:
        Data = './cell_mechanics_2'
        train_img_dir = os.path.join(Data, 'images/train')
        val_img_dir = os.path.join(Data, 'images/val')
        train_label_dir = os.path.join(Data, 'labels/train')
        val_label_dir = os.path.join(Data, 'labels/val')

        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(val_img_dir, exist_ok=True)
        os.makedirs(val_label_dir, exist_ok=True)

    except Exception as e:
        print(f'创建文件夹时出现错误: {e}')

    train_ratio, val_ratio = split_list
    all_img_files = [f for f in os.listdir(img_path) if f.endswith('.bmp')]
    random.shuffle(all_img_files)

    train_size = int(len(all_img_files) * train_ratio)
    train_img_files = all_img_files[:train_size]
    val_img_files = all_img_files[train_size:]

    def copy_files(file_list, img_dest, label_dest):
        for img_file in tqdm(file_list, unit='img'):
            img_src = os.path.join(img_path, img_file)
            label_name = os.path.splitext(img_file)[0] + '.txt'
            label_src = os.path.join(label_path, label_name)

            if os.path.exists(label_src):
                shutil.copy(img_src, img_dest)
                shutil.copy(label_src, label_dest)
            else:
                print(f"未找到对应的标签文件: {label_src}")

    print("开始复制训练集文件...")
    copy_files(train_img_files, train_img_dir, train_label_dir)
    print("开始复制验证集文件...")
    copy_files(val_img_files, val_img_dir, val_label_dir)


if __name__ == '__main__':
    img_path = 'I:/3/Cell_detection_analysis_2'
    label_path = 'G:/cellular_mechanics/cell_detection_normal'
    split_list = [0.8, 0.2]
    split_img(img_path, label_path, split_list)
    