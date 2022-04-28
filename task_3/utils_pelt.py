import torch
from torchvision import datasets, transforms
import numpy as np
import cv2, os
from PIL import Image
from math import floor

def testings():

    image_root = "/Users/PELLERITO/Desktop/Task3/food/"
    image_file = "00000.jpg"
    image = Image.open(os.path.join(image_root, image_file))
    model = load_backbone()
    out = backbone(image, model)
    print(out)


def load_backbone():
    # load the model from torchvision archives
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
    model.eval()
    return model

def backbone(image_orig, model):
    # try with resnet152

    # standardize the image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image_orig)
    image = image.unsqueeze(0)

    # run inference
    out = model(image)
    return out

def get_correct_shape(image):
    # pad the image
    HEIGHT = 300
    WIDTH = 300
    shape = np.shape(image)
    #print("image shape before ", np.shape(image))
    corrected_image = image
    if (shape[0], shape[1]) == (HEIGHT, WIDTH):
        return image

    elif (shape[0], shape[1]) > (HEIGHT, WIDTH):
        corrected_image = cv2.resize(image, dsize=(1194, 800), interpolation=cv2.INTER_CUBIC)
        # of the image having less then 1194 AND 800 dimension

    elif shape[0] < HEIGHT and shape[1] < WIDTH and shape[0] and (shape[0]%2 == 0 and shape[1]%2 == 0) :

        HEIGHT_pad = floor((HEIGHT - shape[0])/2)
        WIDTH_pad = floor((WIDTH - shape[1])/2)

        corrected_image = cv2.copyMakeBorder(
            corrected_image, HEIGHT_pad, HEIGHT_pad, WIDTH_pad, WIDTH_pad, cv2.BORDER_CONSTANT, value=0)


    elif shape[0] < HEIGHT and shape[1] < WIDTH and (shape[0]%2 == 0 and shape[1]%2 != 0):

        HEIGHT_pad = floor((HEIGHT - shape[0])/2)
        WIDTH_pad = floor((WIDTH - shape[1])/2)

        corrected_image = cv2.copyMakeBorder(
            corrected_image, HEIGHT_pad, HEIGHT_pad, WIDTH_pad+1, WIDTH_pad,
            cv2.BORDER_CONSTANT, value=0)

    elif shape[0] < HEIGHT and shape[1] < WIDTH and (shape[0]%2 != 0 and shape[1]%2 == 0):

        HEIGHT_pad = floor((HEIGHT - shape[0])/2)
        WIDTH_pad = floor((WIDTH - shape[1])/2)

        corrected_image = cv2.copyMakeBorder(
            corrected_image, HEIGHT_pad+1, HEIGHT_pad, WIDTH_pad, WIDTH_pad,
            cv2.BORDER_CONSTANT, value=0)

    #print("image shape afterward ", np.shape(corrected_image))
    return corrected_image