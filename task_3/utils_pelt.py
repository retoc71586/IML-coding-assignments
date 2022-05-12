import torch
from torchvision import datasets, transforms
import numpy as np
import cv2, os, tqdm, time
from tqdm import tqdm
from os import listdir
from PIL import Image
from math import floor
import utils_manzo
import numpy as np


def testings():
    image_root = "./food/"
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


def backbone():
    # returns np array features containing all the 10000 extracted features from images

    pwd = os.getcwd()
    features_path = pwd + '/my_features.csv'
    print('Features path:', features_path)
    # try with resnet152
    model = load_backbone()
    # NB: the model outputs one single row feature for each image

    # standardize the image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if os.path.exists(features_path):
        print('Features already extracted. Loading them from', features_path, '!')
        # Load features
        data = np.genfromtxt('my_features.csv', delimiter=",")
        features = np.array(data)
    else:
        # Will contain the feature
        features = []
        # Iterate each image
        curr_dir = os.getcwd()
        # Set the image path
        image_root = curr_dir + "/food"
        for image_file in tqdm(os.listdir(image_root), desc='Extracting features from food images'):
            if image_file.endswith(".jpg"):
                # Read the file
                img = Image.open(os.path.join(image_root, image_file))
                # Transform the image
                image = preprocess(img)
                image = image.unsqueeze(0)
                out = model(image)
                np_out = out.detach().numpy()
                # Convert to NumPy Array, Reshape it, and save it to features variable
                features.append(np_out)
        features = np.array(features).squeeze(1)
        utils_manzo.saveFeatures(features)

    print('Features contains', np.shape(features), 'elements')
    return features


def get_correct_shape(image):
    # pad the image
    HEIGHT = 300
    WIDTH = 300
    shape = np.shape(image)
    # print("image shape before ", np.shape(image))
    corrected_image = image
    if (shape[0], shape[1]) == (HEIGHT, WIDTH):
        return image

    elif (shape[0], shape[1]) > (HEIGHT, WIDTH):
        corrected_image = cv2.resize(image, dsize=(1194, 800), interpolation=cv2.INTER_CUBIC)
        # of the image having less then 1194 AND 800 dimension

    elif shape[0] < HEIGHT and shape[1] < WIDTH and shape[0] and (shape[0] % 2 == 0 and shape[1] % 2 == 0):

        HEIGHT_pad = floor((HEIGHT - shape[0]) / 2)
        WIDTH_pad = floor((WIDTH - shape[1]) / 2)

        corrected_image = cv2.copyMakeBorder(
            corrected_image, HEIGHT_pad, HEIGHT_pad, WIDTH_pad, WIDTH_pad, cv2.BORDER_CONSTANT, value=0)


    elif shape[0] < HEIGHT and shape[1] < WIDTH and (shape[0] % 2 == 0 and shape[1] % 2 != 0):

        HEIGHT_pad = floor((HEIGHT - shape[0]) / 2)
        WIDTH_pad = floor((WIDTH - shape[1]) / 2)

        corrected_image = cv2.copyMakeBorder(
            corrected_image, HEIGHT_pad, HEIGHT_pad, WIDTH_pad + 1, WIDTH_pad,
            cv2.BORDER_CONSTANT, value=0)

    elif shape[0] < HEIGHT and shape[1] < WIDTH and (shape[0] % 2 != 0 and shape[1] % 2 == 0):

        HEIGHT_pad = floor((HEIGHT - shape[0]) / 2)
        WIDTH_pad = floor((WIDTH - shape[1]) / 2)

        corrected_image = cv2.copyMakeBorder(
            corrected_image, HEIGHT_pad + 1, HEIGHT_pad, WIDTH_pad, WIDTH_pad,
            cv2.BORDER_CONSTANT, value=0)

    # print("image shape afterward ", np.shape(corrected_image))
    return corrected_image
