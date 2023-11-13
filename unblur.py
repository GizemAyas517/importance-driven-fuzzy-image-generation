import cv2
import numpy as np
from google.colab.patches import cv2_imshow

image_path = EXPERIMENT_DATA_ARGS[experiment_type]["image_path"]

image = cv2.imread(image_path)

# Create ROI coordinates
topLeft = (60, 140)
bottomRight = (340, 350)
x, y = topLeft[0], topLeft[1]
w, h = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]

# Grab ROI with Numpy slicing and blur
ROI = image[y:y+h, x:x+w]
blur = cv2.GaussianBlur(ROI, (51,51), 0) 

# Insert ROI back into image
image[y:y+h, x:x+w] = blur

cv2_imshow(blur)
cv2_imshow(image)
filename="blurred.jpg"
cv2.imwrite(filename, image)

def get_coupled_results(result_batch, transformed_image):
    result_tensors = result_batch[0]  # there's one image in our batch
    final_rec = tensor2im(result_tensors[-1]).resize(resize_amount)
    input_im = tensor2im(transformed_image).resize(resize_amount)
    res = np.concatenate([np.array(input_im), np.array(final_rec)], axis=1)
    res = Image.fromarray(res)
    return res

if opts.dataset_type == "cars":
    resize_amount = (256, 192) if opts.resize_outputs else (512, 384)
else:
    resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)

res = get_coupled_results(result_batch, transformed_image)
res

if opts.dataset_type == "cars":
    resize_amount = (256, 192) if opts.resize_outputs else (512, 384)
else:
    resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)

res = get_coupled_results(result_batch, transformed_image)
res