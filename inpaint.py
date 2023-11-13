#@title Run this sell to set everything up
print('\n> Cloning the repo')
!git clone https://github.com/saic-mdal/lama.git

print('\n> Install dependencies')
!pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 torchtext==0.9
!pip install -r lama/requirements.txt --quiet
!pip install wget --quiet
!pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html --quiet


print('\n> Changing the dir to:')
%cd /content/lama

print('\n> Download the model')
!curl -L $(yadisk-direct https://disk.yandex.ru/d/ouP6l8VJ0HpMZg) -o big-lama.zip
!unzip big-lama.zip

print('>fixing opencv')
!pip uninstall opencv-python-headless -y --quiet
!pip install opencv-python-headless==4.1.2.30 --quiet


print('\n> Init mask-drawing code')
import base64, os
from IPython.display import HTML, Image
from google.colab.output import eval_js
from base64 import b64decode
import matplotlib.pyplot as plt
import numpy as np
import wget
from shutil import copyfile
import shutil



canvas_html = ""

def draw(imgm, filename='drawing.png', w=400, h=200, line_width=1):
  display(HTML(canvas_html % (w, h, w,h, filename.split('.')[-1], imgm, line_width)))
  data = eval_js("data")
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)


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
  
for i in range(1000,2000):
  masked_src = '/content/drive/MyDrive/whole_masked/'+str(i)+'.jpg'
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

    plt.rcParams['figure.dpi'] = 200
    plt.imshow(plt.imread(f"/content/output/{fname.split('.')[1].split('/')[2]}_mask.png"))
    _=plt.axis('off')
    _=plt.title('inpainting result')
    #plt.show()
    inpainted_im = cv2.imread(f"/content/output/{fname.split('.')[1].split('/')[2]}_mask.png")
    cv2.imwrite("/content/drive/MyDrive/inpaint_result/"+img_name+'.png', inpainted_im)
    fname = None