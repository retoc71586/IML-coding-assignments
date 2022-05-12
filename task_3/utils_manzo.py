import os
from zipfile import ZipFile
import pandas as pd
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Model, Input
from keras.layers import Activation
from keras.legacy_tf_layers.core import Dense, Dropout
from tensorflow import keras
import csv
import utils_tf
import os
import pickle
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

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
        if gen_labels:
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


def checkIllConditioning(input_array):
    num_zeros = np.count_nonzero(input_array == 0)
    num_ones = np.count_nonzero(input_array == 1)
    if num_zeros > 0:
        print(' WARNING:', input_array, 'contains :', num_zeros, '0 elements')
    if num_ones > 0:
        print(' WARNING:', input_array, 'contains :', num_ones, '1 elements')


def plotGraphs(plot_vec, plot_name):
    x_axis = np.linspace(0, len(plot_vec), num=len(plot_vec))
    plt.plot(x_axis, plot_vec)
    plt.ylabel(plot_name)
    # plt.show()
    if os.path.isfile(plot_name + '_plot.png'):
        os.remove(plot_name + '_plot.png')
    plt.savefig(plot_name + '_plot.png')
    # print(plot_name + ' graph plotted!')


def trainModelTorch(TinyModel, features):
    # Select device on which we will train our model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Takes the np array features. Extract features triplets following train_triplets.txt
    # Successively trains the model on them and outputs the weights in classification_net.pth
    criterion = torch.nn.BCELoss(reduction='sum')
    optimizer = optim.Adam(TinyModel.parameters(), lr=1e-4, amsgrad=True)

    # Get train tensors and labels
    print("Generating training and test features tensor...")
    train_tensors, labels = extractFeaturesTriplets(features, 'train_triplets.txt', gen_labels=True)

    # Print some statistics
    # print('Size of train_tensors: ', train_tensors.shape, '. Twice the number of train_triplets, half correct --> '
    #                                                      'label 1, half switched --> label 0')
    # print('Size of test_tensors: ', test_tensors.shape)
    # print('Size of labels: ', labels.shape)

    print("Feature tensors generated!")

    running_loss = 0.0
    running_accuracy = 0.0
    loss_vec = []
    accuracy_vec = []

    for epoch in range(40):
        i = 0
        if os.path.isfile('./checkpoints/model.pt'):
            checkpoint = torch.load('./checkpoints/model.pt', map_location=device)
            TinyModel.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            loss = checkpoint['loss']

            TinyModel = TinyModel.to(device)

        for batch in createMiniBatches(train_tensors, labels, 10, shuffle=True):
            # get the inputs; data is a list of [inputs, labels]
            curr_input, curr_label = batch
            checkIllConditioning(curr_input)

            # Conversion from numpy to torch tensors
            curr_label = torch.tensor(curr_label, dtype=torch.double).unsqueeze(-1).to(device)
            curr_input = torch.tensor(curr_input, dtype=torch.double, requires_grad=True).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            TinyModel.train()
            output = TinyModel(curr_input)
            # input("Press Enter to continue...")

            loss = criterion(output, curr_label)
            loss.backward()
            optimizer.step()

            # Print statistics:
            # print('output shape: ', output.shape, 'label shape: ', curr_label.shape)
            # --> accuracy:
            reshaped_output = output.reshape(-1).cpu().detach().numpy()
            reshaped_label = curr_label.reshape(-1).cpu().detach().numpy()
            acc = (reshaped_output.round() == reshaped_label).mean()

            running_loss += loss.item()
            running_accuracy += acc

            loss_vec.append(loss.item())
            accuracy_vec.append(acc)

            if (i % 2000 == 0) and (i != 0):  # print every 2000 mini-batches
                print(
                    f'[epoch: {epoch + 1}, iteration: {i:5d}] loss: {running_loss / 2000:.3f} accuracy: {running_accuracy / 2000:.3f}')
                running_loss = 0.0
                running_accuracy = 0.0
                print('[output   |   label] :\n',
                      np.c_[output.cpu().detach().numpy(), curr_label.cpu().detach().numpy()])
                # plotGraphs(loss_vec, 'Loss')
                # plotGraphs(accuracy_vec, 'Accuracy')
                # print('Loss and Accuracy graphs plotted')
            i = i + 1
        TinyModel = TinyModel.to('cpu')
        torch.save({
            'model_state_dict': TinyModel.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_vec[-1],
        }, './checkpoints/model.pt')

    print('Finished Training')
    torch.save(TinyModel.state_dict(), './classification_net.pth')


def evaluateModelTorch(TinyModel, features):
    # Select device on which we will train our model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    checkpoint = torch.load('./checkpoints/model.pt', map_location=device)
    TinyModel.load_state_dict(checkpoint['model_state_dict'])
    TinyModel.eval()

    # Get test tensor
    test_tensors = extractFeaturesTriplets(features, 'test_triplets.txt', gen_labels=False)
    test_tensors = torch.tensor(test_tensors, dtype=torch.double).to(device)

    # Inference
    output = TinyModel(test_tensors)

    # Create submission file
    print("Generating submission file...")

    y_test_thresh = output.detach().numpy().round()
    submission_file_path = 'output.txt'
    np.savetxt(submission_file_path, y_test_thresh, fmt='%d')
    print("Submission file generated!")


def pipeline_tensorflow():
    features = utils_tf.get_features()
    print("Generating training and test features tensor...")
    # Get train tensors and labels
    train_tensors, labels = utils_tf.get_triplet_dataset(features, 'train_triplets.txt', gen_labels=True)
    # Get test tensor
    test_tensors = utils_tf.get_triplet_dataset(features, 'test_triplets.txt', gen_labels=False)
    print("Feature tensors generated!")
    trained_model_path = 'veryBigModel'
    if not os.path.exists(trained_model_path):
        print("Building model...")
        # Build model to process features
        x_in = Input(train_tensors.shape[1:])
        x = Activation('leaky_relu')(x_in)
        x = Dropout(0.76)(x)
        x = Dense(1000)(x)
        x = Activation('leaky_relu')(x)
        x = Dense(200)(x)
        x = Activation('leaky_relu')(x)
        x = Dropout(0.74)(x)
        x = Dense(100)(x)
        x = Activation('leaky_relu')(x)
        x = Dense(20)(x)
        x = Activation('leaky_relu')(x)
        x = Dense(1)(x)
        x = Activation('sigmoid')(x)
        model = Model(inputs=x_in, outputs=x)

        print("Train model model...")
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'], )
        epochs = 20
        look_for_valid = False
        if look_for_valid:
            model.fit(x=train_tensors, y=labels, epochs=epochs, batch_size=1, validation_split=0.3, shuffle=1)
        else:
            model.fit(x=train_tensors, y=labels, epochs=epochs)
        print("Training completed!")

        # save trained model
        print("saving model in : ", trained_model_path)
        model.save(trained_model_path)
    else:
        print("your model exists in path : ", trained_model_path)
        model = keras.models.load_model(trained_model_path)

    # Predict
    y_test = model.predict(test_tensors)

    # Create submission file
    y_test_thresh = np.where(y_test < 0.5, 0, 1)
    np.savetxt("submission.txt", y_test_thresh, fmt='%d')
