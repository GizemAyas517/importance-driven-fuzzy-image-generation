"""
To run this file please clone the LAMA repository

git clone https://github.com/saic-mdal/lama.git


Change your directory to lama

cd /content/lama

Download the trained model

!curl -L $(yadisk-direct https://disk.yandex.ru/d/ouP6l8VJ0HpMZg) -o big-lama.zip
!unzip big-lama.zip

"""

import cv2
from os.path import exists

import torch
import torchvision.transforms as transforms
import torchvision

def get_number_for_ds(num):
  if num > 0 and num < 10:
    return "0000"+str(num)
  elif num >=10 and num < 100:
    return "000"+str(num)
  elif num >= 100 and num < 1000:
    return "00"+str(num)
  elif num >=1000 and num < 10000:
    return "0"+str(num)
  else:
    return str(num)
  

def inpaint_images_with_lama(dataset_size, source_patched_images_folder, destination_folder):
    """
        Inpainting code is mostly taken from https://github.com/advimman/lama .
    """
    for i in range(dataset_size):
        masked_src = source_patched_images_folder+str(i)+'.jpg'
        file_exists = exists(masked_src)
        if file_exists:
            img_name = get_number_for_ds(i+1)
            fname = './data_for_prediction/'+img_name+'.jpg'

            print('Run inpainting')
            if '.jpeg' in fname:
                !PYTHONPATH=. TORCH_HOME=$(pwd) python3 bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/data_for_prediction outdir=/content/output dataset.img_suffix=.jpeg > /dev/null
            elif '.jpg' in fname:
                !PYTHONPATH=. TORCH_HOME=$(pwd) python3 bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/data_for_prediction outdir=/content/output  dataset.img_suffix=.jpg > /dev/null
            elif '.png' in fname:
                !PYTHONPATH=. TORCH_HOME=$(pwd) python3 bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/data_for_prediction outdir=/content/output  dataset.img_suffix=.png > /dev/null
            else:
                print(f'Error: unknown suffix .{fname.split(".")[-1]} use [.png, .jpeg, .jpg]')

            inpainted_im = cv2.imread(f"/content/output/{fname.split('.')[1].split('/')[2]}_mask.png")
            cv2.imwrite(destination_folder+img_name+'.png', inpainted_im)
            fname = None

transform = transforms.Compose(
    [transforms.Resize((256, 256)),
    transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.StanfordCars(root='./data', split='train',
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)


dataset_size = len(trainset)
source_patched_images_folder = ""
destination_folder = ""
inpaint_images_with_lama(dataset_size, source_patched_images_folder, destination_folder)