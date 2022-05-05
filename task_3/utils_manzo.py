import os
from zipfile import ZipFile
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn

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
    # for feature in tqdm(features_np, desc='Storing extracted features: '):
    print('Storing features in my_features.csv!\n')
    with open('my_features.csv', 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(features_np)


def extractFeaturesTriplets(features, triplets_file, gen_labels=False):
    # Import pandas
    triplets_df = pd.read_csv(triplets_file, delim_whitespace=True, header=None, names=["Q", "P", "N"])
    # Features tensor
    train_tensors = []
    # Labels
    labels = []
    # Number of triplets in the file
    num_triplets = len(triplets_df)

    for i in range(num_triplets):
        # Get triplet
        triplet = triplets_df.iloc[i]
        Q, P, N = triplet['Q'], triplet['P'], triplet['N']
        # Get features
        tensor_q = features[Q]
        tensor_p = features[P]
        tensor_n = features[N]
        # Concatenete
        triplet_tensor = np.concatenate((tensor_q, tensor_p, tensor_n), axis=-1)
        if (gen_labels):
            reverse_triplet_tensor = np.concatenate((tensor_q, tensor_n, tensor_p), axis=-1)
            # Add to train tensors
            train_tensors.append(triplet_tensor)
            labels.append(1)
            train_tensors.append(reverse_triplet_tensor)
            labels.append(0)
        else:
            train_tensors.append(triplet_tensor)

    train_tensors = np.array(train_tensors)
    if (gen_labels):
        labels = np.array(labels)
        return train_tensors, labels
    else:
        return train_tensors


def createMiniBatches(inputs, labels, batchsize, shuffle=True):
    assert inputs.shape[0] == labels.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], labels[excerpt]


def trainModel(TinyModel, features):
    # Takes the np array features. Extract features triplets following train_triplets.txt
    # Successively trains the model on them and outputs the weights in classification_net.pth
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(TinyModel.parameters(), lr=1e-5)

    triplets_file = os.getcwd() + '/train_triplets.txt'

    print("Generating training and test features tensor...")
    # Get train tensors and labels
    train_tensors, labels = extractFeaturesTriplets(features, 'train_triplets.txt', gen_labels=True)
    # Get test tensor
    test_tensors = extractFeaturesTriplets(features, 'test_triplets.txt', gen_labels=False)
    print('Size of train_tensors: ', train_tensors.shape, '. Twice the number of train_triplets, half correct --> '
                                                          'label 1, half switched --> label 0')
    print('Size of test_tensors: ', test_tensors.shape)
    print('Size of labels: ', labels.shape)
    print("Feature tensors generated!")

    for epoch in range(4):
        i = 0
        for batch in createMiniBatches(train_tensors, labels, 10, shuffle=True):
            x_batch, y_batch = batch
            running_loss = 0.0
            # get the inputs; data is a list of [inputs, labels]
            # curr_label = labels[i]
            curr_label = y_batch
            curr_input = x_batch
            # Conversion from numpy to torch tensors
            curr_label = torch.tensor(curr_label, dtype=torch.double).unsqueeze(-1)
            curr_input = torch.tensor(curr_input, dtype=torch.double)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = TinyModel(curr_input)
            # print('output size: ', output.shape)
            # gitprint('lable size:', curr_label.shape)
            # print('curr_lable: ', curr_label)
            loss = criterion(output, curr_label)
            loss.backward()
            optimizer.step()
            # print('output shape: ',output.shape, 'label shape: ', curr_label.shape)

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[epoch: {epoch + 1}, iteration: {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
                print('output: ', output, 'label: ', curr_label)
            i = i + 1

    print('Finished Training')
    PATH = './classification_net.pth'
    torch.save(TinyModel.state_dict(), PATH)
