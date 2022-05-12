import os
import pickle
import random
import numpy as np
import pandas as pd
from zipfile import ZipFile
# Machine learning libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Lambda


## Execution parameters
# Shape of the resized images
img_shape = [299,299]
# Shape of the input tensor
input_shape = (299,299,3)
# Validation triplets
num_val_triplets = 1500
# Number of images used in the training set
num_images_training = 5000

## Random seed
# Initialize random seed
seed = 528
# Fix random seeds
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

images_archive = 'food.zip'
# Training set
train_triplets_file = 'train_triplets.txt'
# Test set
test_triplets_file = "test_triplets.txt"


def get_features():

    # Extract files from food zip
    zip_ref = ZipFile('food.zip', 'r')
    zip_ref.extractall()
    img_dir = zip_ref.filename[:-4]
    zip_ref.close()

    # get image resized from food file
    res_img_dir = './resized_images'
    if not os.path.exists(res_img_dir):
        os.makedirs(res_img_dir)
        count = 0
        for filename in os.listdir(img_dir):
            count += 1
            print('Processed files: {}/{}'.format(count, len(os.listdir(img_dir))), end="\r")

            if filename.endswith('.jpg'):
                img = img_to_array(load_img(img_dir + '/' + filename))
                img = tf.image.resize_with_pad(img, img_shape[0], img_shape[1], antialias=True)
                img = array_to_img(img)
                img.save(res_img_dir + '/' + str(int(os.path.splitext(filename)[0])) + '.jpg')

        print('All images were successfully resized and saved to disk!')

    features_file = 'features.pckl'
    if not os.path.exists(features_file):

        # Declare feature selection model
        model = backbone(input_shape)

        # Initialize generator
        res_imgs = image_from_directory_generator(res_img_dir, 1)

        # Extract features with inception resnet
        print("Extracting features with pretrained model...")
        features = model.predict(res_imgs, steps=10000)
        # Save features in feature file
        with open(features_file, 'wb') as f:
            pickle.dump(features, f)

    else:
        with open(features_file, 'rb') as f:
            features = pickle.load(f)

    print("Features loaded!")
    return features

# backbone of our network to extract features
def backbone(input_shape):

    # resnet feature extraction
    #resnet_inception = tf.keras.applications.InceptionResNetV2(pooling='avg',include_top=False)

    resnet_inception = tf.keras.applications.VGG16(pooling='avg', include_top=False)
    # restnet takes care of features extraction
    resnet_inception.trainable = False

    # Declare input
    x_in = Input(shape=input_shape)
    x = resnet_inception(x_in)

    # Get the whole model
    model = Model(inputs=x_in, outputs=x)

    return model

# Image generator
def image_from_directory_generator(directory_name, batch_size):
    # Image indices
    num_images = 10000
    # Current idx
    curr_idx = 0

    while True:
        batch = []

        while len(batch) < batch_size:
            img_name = directory_name + "/" + str(int(curr_idx)) + ".jpg"
            img = load_img(img_name)
            img = tf.keras.applications.inception_resnet_v2.preprocess_input(img_to_array(img))
            batch.append(img)
            curr_idx = (curr_idx + 1) % num_images

        batch = np.array(batch)
        labels = np.zeros(batch_size)

        try:
            yield batch, labels
        except StopIteration:
            return

# Create the input tensor of features
def get_triplet_dataset(features, triplets_file, gen_labels=False):
    # Import pandas
    df = pd.read_csv(triplets_file, delim_whitespace=True, header=None, names=["Q", "P", "N"])
    # Features tensor
    train_tensors = []
    # Labels
    labels = []
    # Number of triplets in the file
    num_triplets = len(df)

    for i in range(num_triplets):
        # Get triplet
        triplet = df.iloc[i]
        Q, P, N = triplet['Q'], triplet['P'], triplet['N']
        # Get features
        tensor_q = features[Q]
        tensor_p = features[P]
        tensor_n = features[N]
        # Concatenete
        triplet_tensor = np.concatenate((tensor_q, tensor_p, tensor_n), axis=-1)
        if(gen_labels):
            reverse_triplet_tensor = np.concatenate((tensor_q, tensor_n, tensor_p), axis=-1)
            # Add to train tensors
            train_tensors.append(triplet_tensor)
            labels.append(1)
            train_tensors.append(reverse_triplet_tensor)
            labels.append(0)
        else:
            train_tensors.append(triplet_tensor)


    train_tensors = np.array(train_tensors)
    if(gen_labels):
        labels = np.array(labels)
        return train_tensors, labels
    else:
        return train_tensors
