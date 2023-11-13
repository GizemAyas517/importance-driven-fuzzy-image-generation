import torchvision.transforms as transforms
import torchvision
import torch

# LOAD STANFORD CARS DATASET

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.StanfordCars(root='./data', split='train',
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.StanfordCars(root='./data', split='test',
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import time

def find_w_h(image_size, x, y, box_size):
  start_w = x-(box_size/2)
  start_h = y-(box_size/2)
  final_w = x+(box_size/2)
  final_h = y+(box_size/2)
  if start_w < 0:
    start_w = x
  if start_h < 0:
    start_h = y
  if final_w > image_size:
    final_w = image_size - x
  
  if final_h > image_size:
    final_h = image_size - y

  return int(start_w), int(start_h), int(final_w), int(final_h)

no_img_ind = []
def generate_blur_important(heatmap_img_src, original_image_src, save_location, ind):
  img=cv2.imread(heatmap_img_src)
  img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  # lower mask (0-10)
  lower_red = np.array([0,50,50])
  upper_red = np.array([10,255,255])
  mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

  # upper mask (170-180)
  lower_red = np.array([170,50,50])
  upper_red = np.array([180,255,255])
  mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

  # join my masks
  #mask = mask0+mask1
  mask = mask0+mask1

  # set my output img to zero everywhere except my mask
  output_img = img.copy()
  output_img[np.where(mask==0)] = 0
  original = output_img
  image = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

  detected = cv2.bitwise_and(original, original, mask=mask)

  # Remove noise
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
  opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

  # Find contours and find total area
  cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]

  # Read image
  im = opening

  params = cv2.SimpleBlobDetector_Params() 
  # Change thresholds
  params.minThreshold = 10
  params.maxThreshold = 200

  # Filter by Area.
  params.filterByArea = False
  params.minArea = 10

  # Filter by Circularity
  params.filterByCircularity = False
  params.minCircularity = 0.1

  # Filter by Convexity
  params.filterByConvexity = False
  params.minConvexity = 0.87

  # Filter by Inertia
  params.filterByInertia = False
  params.minInertiaRatio = 0.01

  # Create a detector with the parameters
  ver = (cv2.__version__).split('.')
  if int(ver[0]) < 3 :
      detector = cv2.SimpleBlobDetector(params)
  else: 
      detector = cv2.SimpleBlobDetector_create(params)

  # Detect blobs.
  keypoints = detector.detect(im)
  im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  cv2.imwrite("result.png",im_with_keypoints)
  
  pts = cv2.KeyPoint_convert(keypoints)
  resized = None
  if len(pts) == 0:
    no_img_ind.append(ind)
    print("no pts "+str(ind))
  else:
    try:
      my_im = cv2.imread(original_image_src)
      dim = (288, 288)
      #resized = cv2.resize(res, dsize=(288, 288), interpolation=cv2.INTER_CUBIC)
      resized = cv2.resize(my_im, dim, interpolation = cv2.INTER_AREA)
      for i in range(len(pts)):
        # Create ROI coordinates
        x, y = int(pts[i][0]), int(pts[i][1])
        w1, h1, w2, h2 = find_w_h(288, x, y, 50)

        # Grab ROI with Numpy slicing and blur
        ROI = resized[h1:h2, w1:w2]
        blur = cv2.GaussianBlur(ROI, (51,51), 0) 

        # Insert ROI back into image
        resized[h1:h2, w1:w2] = blur

    # cv2_imshow(resized)
      filename="blurred"
      cv2.imwrite(save_location+filename+str(ind)+".jpg", resized)
    except:
      no_img_ind.append(ind)
      print("error for "+str(ind))


#original_image_src = '/content/drive/MyDrive/car_imgs/class_car/00001.jpg'
#my_im = cv2.imread(original_image_src)
dim = (288, 288)
blank_image = np.zeros([288,288,3],dtype=np.uint8)
#blank_image.fill(255) # or img[:] = 255
cv2.imwrite("blank.jpg", blank_image)
resized = cv2.resize(blank_image, dim, interpolation = cv2.INTER_AREA)
w1, h1, w2, h2 = find_w_h(288, 100, 100, 50)
cv2.rectangle(resized, (w1, h1), (w2, h2), (255,255,255), -1)
save_location='/content/out/'
filename="masked"
cv2.imwrite("gizem.jpg", resized)
