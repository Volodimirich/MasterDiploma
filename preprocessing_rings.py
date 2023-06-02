from tqdm import tqdm
import os
from pathlib import Path
import glob
import numpy as np
import argparse
# Get fourier batches
from PIL import Image

def create_circular(h, w, max_numb=None, center=None, tolerance=1, min_val=0):
    if center is None:  # use the middle of the image
        center = (w / 2 - 0.5, h / 2 - 0.5)
    if max_numb is None:
        max_numb = max(h//2, w//2)

    dist_from_center = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            dist_from_center[i][j] = int(max(abs(i - center[0]), abs(j-center[1]))//tolerance + min_val)
    return dist_from_center

def normalization(value):
    value -= value.min()
    value /= value.max()
    return value

def split_array(imgs, split_coefs=None):
    if split_coefs is None:
        split_coefs = [1, 2, 4]
    fin_res_am = []
    fin_res_ph = []
    
    *_, h, w = imgs.shape
    for coef in split_coefs:
        row_data_amp = []
        row_data_ph = []
        step = h // coef
        # add fft
        try:
            for y in range(0, w, step):
                result_amp = []
                result_ph = []
                for x in range(0, h, step):
                    fft_data = np.fft.fftshift(np.fft.fft2(imgs[..., x:x + step, y:y + step]))
                    amp_val, ph_val = np.abs(fft_data), np.angle(fft_data)
                    amp_val = np.log(1 +  amp_val) #??
                    amp_val, ph_val = normalization(amp_val), normalization(ph_val)
                    result_amp.append(amp_val)
                    result_ph.append(ph_val)

                row_data_amp.append(np.concatenate(result_amp, axis=1))
                row_data_ph.append(np.concatenate(result_ph, axis=1))
        except:
            print(x, y)
        fin_res_am.append(np.concatenate(row_data_amp, axis=0)) 
        fin_res_ph.append(np.concatenate(row_data_ph, axis=0)) 
    return fin_res_am, fin_res_ph

# Get fourier batches 
def split_tensor(imgs, split_coefs=None):
    if split_coefs is None:
        split_coefs = [1, 2, 4]
    result = []

    *_, h, w = imgs.shape
    last_val = 0
    for coef in split_coefs:
        step = h // coef
        # add fft
        row_data = []
        for y in range(0, w, step):
            step_result = []
            for x in range(0, h, step):
                #temperory unused
                value = imgs[x:x + step, y:y + step]
                mask = create_circular(step, step, min_val=last_val)
                last_val = mask[-1][-1] + 1
                                
                
                step_result.append(mask)
            row_data.append(np.concatenate(step_result, axis=1))
        result.append(np.concatenate(row_data, axis=0)) 
    return result



bad_files = ['/raid/data/cats_dogs_dataset/PetImages/Cat/666.jpg', '/raid/data/cats_dogs_dataset/PetImages/Cat/Thumbs.db',
            '/raid/data/cats_dogs_dataset/PetImages/Dog/Thumbs.db', '/raid/data/cats_dogs_dataset/PetImages/Dog/11702.jpg',]

def dataset_preprocessing(directory, type_list, target_dirname, dataset, img_shape=(256, 256), common_path='/raid/data'):
    # if not os.path.exists(f'/raid/data/cats_dogs_dataset/{target_dirname}'):
        # os.mkdir(f'/raid/data/cats_dogs_dataset/{target_dirname}')
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
            if not os.path.exists(f'/raid/data/chest_xray/{target_dirname}/{dir_name}'):
                os.mkdir(f'/raid/data/chest_xray/{target_dirname}/{dir_name}')
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