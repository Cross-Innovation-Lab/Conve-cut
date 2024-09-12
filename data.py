# -*- coding: utf-8 -*-            
# @Author : BingYu Nan
# @Location : Wuxi
# @Time : 2024/8/22 15:07
import argparse
import os
import cv2
import h5py
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path",
                    default='./data/rafdb',
                    type=str,
                    help="dataset path")
parser.add_argument("--dataset_name",
                    default='RAFDB',
                    type=str,
                    help="dataset name")
def load_data(
        path_prefix,
        dataset_name,
        splits=['train', 'val', 'test'],
):
    X, y = {}, {}

    IMG_SIZE = 224 if 'RAFDB' in dataset_name else 120
    splits = ['train', 'test'] if 'RAFDB' in dataset_name else splits
    classNames = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness',
                  'surprise'] if 'RAFDB' in dataset_name else ['anger', 'contempt', 'disgust', 'fear', 'happiness',
                                                               'neutral', 'sadness', 'surprise']

    for split in splits:
        PATH = os.path.join(path_prefix, split)
        print(PATH)
        X[split], y[split] = [], []
        for classes in os.listdir(PATH):
            class_path = os.path.join(PATH, classes)
            print(class_path)
            class_numeric = classNames.index(classes)
            for sample in os.listdir(class_path):
                sample_path = os.path.join(class_path, sample)
                image = cv2.imread(sample_path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                X[split].append(image)
                y[split].append(class_numeric)

    # Convert to numpy arrays
    for split in splits:
        X[split] = np.array(X[split])
        y[split] = np.array(y[split])

    return X, y

def main(dateset_path, dataset_name):
    X, y = load_data(dateset_path, dataset_name=dataset_name)
    with h5py.File(fr'.\{dataset_name}.h5', 'w') as dataset:
        for split in X.keys():
            dataset.create_dataset(f'X_{split}', data=X[split])
            dataset.create_dataset(f'y_{split}', data=y[split])
    del X, y
if __name__ == "__main__":
    args = parser.parse_args()
    dataset_path = args.dataset_path
    dataset_name = args.dataset_name
    print('dataset path:{}'.format(dataset_path))
    print('dataset name:{}'.format(dataset_name))
    main(dataset_path, dataset_name)
