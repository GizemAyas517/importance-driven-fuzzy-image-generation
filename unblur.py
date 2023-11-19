import time
import os
from PIL import Image
import torch
import torchvision.transforms as transforms

from utils.common import tensor2im
from utils.inference_utils import run_inversion
from utils.model_utils import load_model


"""
    Hyperstyle code is taken from https://github.com/yuval-alaluf/hyperstyle

    Clone the repository from https://github.com/yuval-alaluf/hyperstyle.git
"""

def unblur_with_hyperstyle(dataset_path, destination_path):
  EXPERIMENT_DATA_ARGS = {
      "cars": {
        "model_path": path_to_model,
        "w_encoder_path": path_to_w_encoder,
        "image_path": dataset_path,
        "transform": transforms.Compose([
            transforms.Resize((192, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        }
    }

  experiment_type = "cars"
  EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]

  if not os.path.exists(EXPERIMENT_ARGS['model_path']) or os.path.getsize(EXPERIMENT_ARGS['model_path']) < 1000000:
    """
        download this file
        downloader.download_file(file_id=HYPERSTYLE_PATHS[experiment_type]['id'], file_name=HYPERSTYLE_PATHS[experiment_type]['name'])
    """
    # if google drive receives too many requests, we'll reach the quota limit and be unable to download the model
    if os.path.getsize(EXPERIMENT_ARGS['model_path']) < 1000000:
        raise ValueError("Pretrained model was unable to be downloaded correctly!")
    else:
        print('Done.')
  else:
        print(f'HyperStyle model for {experiment_type} already exists!')


  model_path = EXPERIMENT_ARGS['model_path']
  net, opts = load_model(model_path, update_opts={"w_encoder_checkpoint_path": EXPERIMENT_ARGS['w_encoder_path']})
  print('Model successfully loaded!')

  for i in range(8144):
    image_path = EXPERIMENT_DATA_ARGS[experiment_type]["image_path"]+str(i)+".jpg"
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

        outputs_path = destination_path+str(i)+".jpg"
        res.save(outputs_path)


def get_unblurred(result_batch, transformed_image):
    result_tensors = result_batch[0]  # there's one image in our batch
    final_rec = tensor2im(result_tensors[-1]).resize(resize_amount)
    return final_rec


path_to_model = "/content/.../hyperstyle_cars.pt"
path_to_w_encoder = "/content/.../cars_w_encoder.pt"

whole_blurred_dataset_path = "/content/.../whole_blurred/"
whole_blurred_destination = "/content/.../whole_unblurred/"

unblur_with_hyperstyle(whole_blurred_dataset_path, whole_blurred_destination)

important_blurred_dataset_path = "/content/.../important_blurred/"
important_blurred_destination = "/content/.../important_unblurred/"
unblur_with_hyperstyle(important_blurred_dataset_path, important_blurred_destination)