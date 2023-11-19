import time
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

sys.path.append(".")
sys.path.append("..")

from utils.common import tensor2im
from utils.inference_utils import run_inversion
from utils.model_utils import load_model


EXPERIMENT_DATA_ARGS = {
    "cars": {
        "model_path": "/content/drive/MyDrive/hyperstyle_cars.pt",
        "w_encoder_path": "/content/drive/MyDrive/cars_w_encoder.pt",
        "image_path": "/content/drive/MyDrive/whole_blurred/",
        "transform": transforms.Compose([
            transforms.Resize((192, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }
}

#@title Load HyperStyle Model { display-mode: "form" } 
model_path = EXPERIMENT_ARGS['model_path']
net, opts = load_model(model_path, update_opts={"w_encoder_checkpoint_path": EXPERIMENT_ARGS['w_encoder_path']})
print('Model successfully loaded!')
pprint.pprint(vars(opts))


def get_coupled_results(result_batch, transformed_image):
    result_tensors = result_batch[0]  # there's one image in our batch
    final_rec = tensor2im(result_tensors[-1]).resize(resize_amount)
    input_im = tensor2im(transformed_image).resize(resize_amount)
    res = np.concatenate([np.array(input_im), np.array(final_rec)], axis=1)
    res = Image.fromarray(res)
    return res

def get_unblurred(result_batch, transformed_image):
    result_tensors = result_batch[0]  # there's one image in our batch
    final_rec = tensor2im(result_tensors[-1]).resize(resize_amount)
    return final_rec

# unblur whole image

for i in range(8144):
  image_path = EXPERIMENT_DATA_ARGS[experiment_type]["image_path"]+"whole_blurred"+str(i)+".jpg"
  original_image = Image.open(image_path).convert("RGB")
  original_image = original_image.resize((192, 256))

  input_image = original_image
  input_image.resize((256, 256))


  n_iters_per_batch = 5
  opts.n_iters_per_batch = n_iters_per_batch
  opts.resize_outputs = False  # generate outputs at full resolution

  img_transforms = EXPERIMENT_ARGS['transform']
  transformed_image = img_transforms(input_image) 

  with torch.no_grad():
    tic = time.time()
    result_batch, result_latents, _ = run_inversion(transformed_image.unsqueeze(0).cuda(), 
                                                    net, 
                                                    opts,
                                                    return_intermediate_results=True)
    toc = time.time()
    print('Inference took {:.4f} seconds.'.format(toc - tic))
  resize_amount = (256, 192) if opts.resize_outputs else (512, 384)

  res = get_unblurred(result_batch, transformed_image)

  outputs_path = "/content/drive/MyDrive/whole_unblurred/"+str(i)+".jpg"
  res.save(outputs_path)

  # unblur whole_blur_30
for i in range(91, 8144):
#for i in range(8144):
  image_path = EXPERIMENT_DATA_ARGS[experiment_type]["image_path"]+str(i)+".jpg"
  #image_path = filename
  original_image = Image.open(image_path).convert("RGB")
  original_image = original_image.resize((192, 256))

  original_image.resize((256, 256))


  n_iters_per_batch = 5
  opts.n_iters_per_batch = n_iters_per_batch
  opts.resize_outputs = False  # generate outputs at full resolution

  img_transforms = EXPERIMENT_ARGS['transform']
  transformed_image = img_transforms(original_image) 

  with torch.no_grad():
    result_batch, result_latents, _ = run_inversion(transformed_image.unsqueeze(0).cuda(), 
                                                    net, 
                                                    opts,
                                                    return_intermediate_results=True)
  resize_amount = (256, 192) if opts.resize_outputs else (512, 384)

  res = get_unblurred(result_batch, transformed_image)

  outputs_path = "/content/drive/MyDrive/whole_unblurred_30/"+str(i)+".jpg"
  res.save(outputs_path)


  import os.path

EXPERIMENT_DATA_ARGS = {
    "cars": {
        "model_path": "/content/drive/MyDrive/hyperstyle_cars.pt",
        "w_encoder_path": "/content/drive/MyDrive/cars_w_encoder.pt",
        "image_path": "/content/drive/MyDrive/important_blurred/",
        "transform": transforms.Compose([
            transforms.Resize((192, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }
}

EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]

# unblur important neuron blur
for i in range(8144):
  image_path = EXPERIMENT_DATA_ARGS[experiment_type]["image_path"]+"blurred"+str(i)+".jpg"
  if os.path.exists(image_path):
    original_image = Image.open(image_path).convert("RGB")

    original_image = original_image.resize((192, 256))

    original_image.resize((256, 256))


    n_iters_per_batch = 5
    opts.n_iters_per_batch = n_iters_per_batch
    opts.resize_outputs = False  # generate outputs at full resolution

    img_transforms = EXPERIMENT_ARGS['transform']
    transformed_image = img_transforms(original_image) 

    with torch.no_grad():
      result_batch, result_latents, _ = run_inversion(transformed_image.unsqueeze(0).cuda(), 
                                                      net, 
                                                      opts,
                                                      return_intermediate_results=True)
    resize_amount = (256, 192) if opts.resize_outputs else (512, 384)

    res = get_unblurred(result_batch, transformed_image)

    outputs_path = "/content/drive/MyDrive/important_unblurred/"+str(i)+".jpg"
    res.save(outputs_path)