import gc
import numpy as np
import scipy
import sys
from typing import List


import os
import json
import tqdm

from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from collections import Counter

from imbalanced_classification.embedding_debias_methods.INLPMethod import INLPMethod
from imbalanced_classification.embedding_debias_methods.MPMethod import MPMethod, get_directions, get_directions_weighted

from utils.load_biasbios_data import *

def get_prof_data(X, Y, gender, y_value):
    mask = Y == y_value
    filtered_X = X[mask]
    filtered_gender = gender[mask]
    return filtered_X, filtered_gender, mask


def load_debiased_data(datapath, task_name):
    data_folder = f"{datapath}/cleaned_data_{task_name}/"
    print("debiased datafolder: ", data_folder)
    if os.path.exists(data_folder):
        print("Loading existing cleaned data")
        x_train = np.load(data_folder + 'x_train_cleaned.npy')
        x_dev = np.load(data_folder + 'x_dev_cleaned.npy')
        x_test = np.load(data_folder + 'x_test_cleaned.npy')
        return x_train, x_dev, x_test
    return None, None, None

def store_debiased_data(datapath, task_name, cleaned_x_train, cleaned_x_dev, cleaned_x_test):
    data_folder = f"{datapath}/cleaned_data_{task_name}/"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    np.save(data_folder + 'x_train_cleaned.npy', cleaned_x_train)
    np.save(data_folder + 'x_dev_cleaned.npy', cleaned_x_dev)
    np.save(data_folder + 'x_test_cleaned.npy', cleaned_x_test)


def get_inverse_freq(y_train, gender_train):
    from sklearn.preprocessing import LabelEncoder
    import numpy as np

    # Combine y_train and gender_train into a single array
    combined = [f'{y}_{g}' for y, g in zip(y_train, gender_train)]

    # Encode each unique class-gender combination to a unique integer
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(combined)

    # Calculate the frequency of each class-gender combination
    freq = np.bincount(encoded)

    # Take the inverse of the frequencies to get the weights
    input_weights = 1 / freq[encoded]
    return input_weights

def get_MP_cleaned_data_all(x_train, y_train, x_dev, y_dev, x_test, y_test, gender_train, gender_dev, gender_test, config, load_or_store_data=True):
    by_hoc_str = "_posthoc" if config["debiase_posthoc"] else "_prehoc"
    print("DEBIASING DATA USING MP METHOD")
    task_name = config['dataset'] + "_MP_all" + by_hoc_str

    # Get directions
    directions_mp = get_directions(x_train, gender_train)

    mp_object = MPMethod(task_name, directions_mp, input_dim=x_train.shape[-1])
    projection_mp = mp_object.mean_projection_method()

    # Clean data
    cleaned_x_train = np.dot(x_train, projection_mp)  
    cleaned_x_dev = np.dot(x_dev, projection_mp) 
    cleaned_x_test = np.dot(x_test, projection_mp) 

    # Check if there is a row left with only zeros
    if np.any(np.all(cleaned_x_train == 0, axis=1)):
        print("Warning: cleaned_x_train contains a row with only zeros")
    if np.any(np.all(cleaned_x_dev == 0, axis=1)):
        print("Warning: cleaned_x_dev contains a row with only zeros")
    if np.any(np.all(cleaned_x_test == 0, axis=1)):
        print("Warning: cleaned_x_test contains a row with only zeros")

    return cleaned_x_train, cleaned_x_dev, cleaned_x_test



def get_INLP_cleaned_data_all(x_train, y_train, x_dev, y_dev, x_test, y_test, gender_train, gender_dev, gender_test, config, load_or_store_data=True):
    task_name = config['dataset'] + "_INLP_all" 
    datapath = config['datapath']
    verbose = True


    if load_or_store_data and config["debias_load_stored"] is True:
        x_train_stored, x_dev_stored, x_test_stored = load_debiased_data(datapath, task_name)
        if x_train_stored is not None:
            print("Loading existing cleaned data for INLP_total"	)
            return x_train_stored, x_dev_stored, x_test_stored

    model = SGDClassifier
    loss = 'log_loss'
    warm_start = True
    early_stopping = False
    max_iter = 10000

    params = {'warm_start': warm_start, 'loss': loss, 'random_state': 0, 'early_stopping': early_stopping,
              'max_iter': max_iter}


    x_test_shuffled, gender_train_shuffled = shuffle(x_train, gender_train, random_state=0, n_samples=len(x_train))
    # Run INLP method
    inlp_object = INLPMethod(model, params)

    majority = Counter(gender_dev).most_common(1)[0][1] / float(len(gender_dev))

    print('MAJORITY % perc:', majority)

    # Main function for INLP
    output_p, _, _, _ = inlp_object.get_projection_matrix(config["inlp_num_classifiers"],
                                                        x_test_shuffled, 
                                                        gender_train_shuffled,
                                                        x_dev, gender_dev,
                                                        majority_acc=0.1, verbose=verbose, by_class=True, 
                                                        Y_train_main=y_train, Y_dev_main=y_dev)
    # Clean data
    x_train_clean = np.dot(x_train, output_p)  
    x_dev_clean = np.dot(x_dev, output_p) 
    x_test_clean = np.dot(x_test, output_p) 

    # Save cleaned data
    if load_or_store_data:
        store_debiased_data(datapath, task_name, x_train_clean, x_dev_clean, x_test_clean)

    return x_train_clean, x_dev_clean, x_test_clean



def get_debiase_method(config):
    method = config["debias_embs"]
    if method == "mp":
        print("MP - DEBIASING EMBEDDINGS - ALL CLASSES TOGETHER")
        return get_MP_cleaned_data_all
    elif method == "inlp":
        print("INLP - DEBIASING EMBEDDINGS - ALL CLASSES TOGETHER")
        return get_INLP_cleaned_data_all
    else:
        print(f"Unknown method: {method}")
        return None
    

import torch
from torch.utils.data import DataLoader
def get_embeddings(model, dataloader):
    
    embeddings = []

    def hook(module, input, output):
        embeddings.append(output)

    # Register the hook on the last layer of the model
    model[-2].register_forward_hook(hook)

    # Pass the data through the model
    for batch in dataloader:
        batch = batch[0]
        model(batch)

    # Remove the hook
    model[-2]._forward_hooks.clear()

    return torch.cat(embeddings)

def get_hidden_states(model, dataset_train, dataset_dev, dataset_test, num_workers=0):
    model.eval()

    batch_size = 512
    dataloader_train = DataLoader(dataset_train, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    dataloader_dev = DataLoader(dataset_dev, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    dataloader_test = DataLoader(dataset_test, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    new_x_train = get_embeddings(model, dataloader_train)
    new_x_dev = get_embeddings(model, dataloader_dev)
    new_x_test = get_embeddings(model, dataloader_test)

    model.train()

    return new_x_train.cpu().detach().numpy(), new_x_dev.cpu().detach().numpy(), new_x_test.cpu().detach().numpy()
