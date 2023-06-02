import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import os.path as osp
from tqdm import tqdm
from PIL import Image
import argparse
import os

from pathlib import Path
import glob
import random


def create_perc_split(img, amount_of_steps):
    previous = np.percentile(img, 100)
    draw_img = np.zeros_like(img)
    step = 100 / amount_of_steps
    for itt_num in range(0, amount_of_steps):
        border = np.percentile(img, 100 - step * (itt_num + 1))
        result = np.where((previous >= img) & (img > border), itt_num, 0)
        previous = border
        draw_img += result
    return draw_img


def prepare_data(img, mask, amount_of_steps):
    h, w = img.shape
    results = []
    pixel_per_isocline = h * w // amount_of_steps

    for i in range(amount_of_steps):
        line = img[mask == i]
        if line.shape[0] != pixel_per_isocline:
            idx_list = random.sample(range(1, line.shape[0] - 1), abs(pixel_per_isocline - line.shape[0]))
            is_insert = True if line.shape[0] < pixel_per_isocline else False

            if not is_insert:
                line = np.delete(line, idx_list)
            else:
                mean_val = line.mean()
                line = np.insert(line, idx_list, mean_val)
        results.append(line)
    return results


# Get fourier batches
def split_tensor(imgs, split_coefs=None, amount_of_isoc=32):
    if split_coefs is None:
        split_coefs = [1, 2, 4]
    result = []
    result_ph = []

    *_, h, w = imgs.shape
    for coef in split_coefs:
        step = h // coef
        for y in range(0, w, step):
            step_result = []
            for x in range(0, h, step):
                value = np.abs(np.fft.fftshift(np.fft.fft2(imgs[..., x:x + step, y:y + step])))
                phase = np.angle(np.fft.fftshift(np.fft.fft2(imgs[..., x:x + step, y:y + step])))

                value -= value.min()
                value /= value.max()

                phase -= phase.min()
                phase /= phase.max()

                mask = create_perc_split(value, amount_of_steps=amount_of_isoc)
                data = prepare_data(value, mask, amount_of_steps=amount_of_isoc)
                data_ph = prepare_data(phase, mask, amount_of_steps=amount_of_isoc)

                result.append(data)
                result_ph.append(data_ph)
    return result, result_ph


# CAT - 0, DOG - 1


bad_files = ['/raid/data/cats_dogs_dataset/PetImages/Cat/666.jpg',
             '/raid/data/cats_dogs_dataset/PetImages/Cat/Thumbs.db',
             '/raid/data/cats_dogs_dataset/PetImages/Dog/Thumbs.db',
             '/raid/data/cats_dogs_dataset/PetImages/Dog/11702.jpg', ]


# def dataset_preprocessing(directory, type_list=[('Cat', 0), ('Dog', 1)],
                        #   img_shape=(256, 256), target_dirname='preprocessed', dataset = 'cats_dogs_dataset'):

def dataset_preprocessing(directory, type_list, target_dirname, dataset, img_shape=(256, 256), common_path='/raid/data'):
    if not os.path.exists(f'{common_path}/{dataset}/{target_dirname}'):
        os.mkdir(f'{common_path}/{dataset}/{target_dirname}')


    for dir_name, target_type in type_list:
        for file in tqdm(glob.glob(f'{directory}/{dir_name}/*')):
            if file in bad_files:
                continue
            data = np.array(Image.open(file).convert('L').resize(img_shape, Image.ANTIALIAS))
            try:
                data_amp, data_phase = split_tensor(data)
            except:
                bad_files.append(file)
                continue
            if not os.path.exists(f'{common_path}/{dataset}/{target_dirname}/{dir_name}'):
                os.mkdir(f'{common_path}/{dataset}/{target_dirname}/{dir_name}')
            np.save(f'{common_path}/{dataset}/{target_dirname}/{dir_name}/{Path(file).stem}.npy',
                    [data_amp, data_phase, target_type])
                       


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset preprocessing')
    parser.add_argument('mode', metavar='m', type=int, nargs='+',
                        help='Preprocessing mode (chest/cats)', default='chest')
    args = parser.parse_args()

    if args.mode == 'chest':
        type_list = [('NORMAL', 0), ('PNEUMONIA', 1)]
        dataset_preprocessing('/raid/data/chest_xray/train', type_list,
                            'preprocessed_xray', 'chest_xray')
    elif args.mode == 'cats':
        type_list = [('Cat', 0), ('Dog', 1)]
        dataset_preprocessing('/raid/data/cats_dogs_dataset/train', type_list,
                            'preprocessed', 'cats_dogs_dataset')
    else:
        print('Wrong modes, waited chest or cats. Exit')