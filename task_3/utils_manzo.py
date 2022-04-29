import os
from zipfile import ZipFile

import numpy as np
from PIL import Image
from tqdm import tqdm
import csv

def unzip(zip_file):
    pwd = os.getcwd()
    if not os.path.exists(pwd + '/food'):
        print('Extracting .zip archive...')
        # Open .zip archive
        zip_ref = ZipFile(zip_file, 'r')
        # Extract food.zip to the zip directory
        zip_ref.extractall()
        # Get zip folder name
        img_dir = zip_ref.filename[:-4]
        # Close .zip archive
        zip_ref.close()
        # Only debug purposes
        print('Archive successfully extracted!')
    else:
        print('Food zip archive already extracted!')

def saveFeatures(features_np):
    pwd = os.getcwd()
    #for feature in tqdm(features_np, desc='Storing extracted features: '):
    print('Storing features in my_features.csv!\n')
    with open('my_features.csv', 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(features_np)
