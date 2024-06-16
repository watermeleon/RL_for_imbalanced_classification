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

from utils.load_process_data import *

def get_prof_data(X, Y, gender, y_value):
    mask = Y == y_value
    filtered_X = X[mask]
    filtered_Y = Y[mask]
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
    # combined = list(zip(y_train, gender_train))
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
    by_class_str = "_weighted" if config["weighted_mp"] else ""
    by_hoc_str = "_posthoc" if config["debiase_posthoc"] else "_prehoc"
    print("DEBIASING DATA USING MP METHOD")
    task_name = config['dataset'] + "_MP_all" + by_class_str + by_hoc_str
    datapath = config['datapath']

    if load_or_store_data and config["debias_load_stored"] is True:
        x_train_stored, x_dev_stored, x_test_stored = load_debiased_data(datapath, task_name)
        if x_train_stored is not None:
            return x_train_stored, x_dev_stored, x_test_stored

    # Get directions
    if config["weighted_mp"] is True:
        print("--- Using weighted directions ---")
        input_weights =  get_inverse_freq(y_train, gender_train)
        directions_mp = get_directions_weighted(x_train, gender_train, input_weights)
    else:
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

    # Save cleaned data
    if load_or_store_data:
        store_debiased_data(datapath, task_name, cleaned_x_train, cleaned_x_dev, cleaned_x_test)

    return cleaned_x_train, cleaned_x_dev, cleaned_x_test



def get_INLP_cleaned_data_all(x_train, y_train, x_dev, y_dev, x_test, y_test, gender_train, gender_dev, gender_test, config, load_or_store_data=True):
    by_class_str = "_by_class" if config["inlp_by_class"] else ""
    task_name = config['dataset'] + "_INLP_all" + by_class_str
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
    # majority = min(majority, 0.8)
    # majority = 0.67
    majority = 0.1

    # if verbose: print('MAJORITY % perc:', majority)
    print('MAJORITY % perc:', majority)

    # Main function for INLP
    output_p, _, _, _ = inlp_object.get_projection_matrix(config["inlp_num_classifiers"],
                                                        x_test_shuffled, 
                                                        gender_train_shuffled,
                                                        x_dev, gender_dev,
                                                        majority_acc=majority, verbose=verbose, by_class=config["inlp_by_class"], 
                                                        Y_train_main=y_train, Y_dev_main=y_dev)
    # Clean data
    x_train_clean = np.dot(x_train, output_p)  
    x_dev_clean = np.dot(x_dev, output_p) 
    x_test_clean = np.dot(x_test, output_p) 

    # Save cleaned data
    if load_or_store_data:
        store_debiased_data(datapath, task_name, x_train_clean, x_dev_clean, x_test_clean)

    return x_train_clean, x_dev_clean, x_test_clean





def get_MP_cleaned_data_per_class(x_train, y_train, x_dev, y_dev, x_test, y_test, gender_train, gender_dev, gender_test, config, load_or_store_data=True):
    print("DEBIASING DATA USING MP METHOD")
    task_name = config['dataset'] + "_MP_per_class"
    datapath = config['datapath']

    if load_or_store_data and config["debias_load_stored"] is True:
        x_train_stored, x_dev_stored, x_test_stored = load_debiased_data(datapath, task_name)
        if x_train_stored is not None:
            return x_train_stored, x_dev_stored, x_test_stored


    # Get unique professions
    unique_professions = np.unique(y_train)

    # Create copies of the original data
    cleaned_x_train = np.zeros_like(x_train)
    cleaned_x_dev = np.zeros_like(x_dev)
    cleaned_x_test = np.zeros_like(x_test)

    # Process each profession separately
    for prof_idx in tqdm.tqdm(unique_professions):
        # Filter data for current profession and get indices
        filt_x_train, filt_gender_train, train_indices = get_prof_data(x_train, y_train, gender_train, prof_idx)
        filt_x_dev, _, dev_indices = get_prof_data(x_dev, y_dev, gender_dev, prof_idx)
        filt_x_test, _, test_indices = get_prof_data(x_test, y_test, gender_test, prof_idx)

        # Get directions
        directions_mp = get_directions(filt_x_train, filt_gender_train)

        mp_object = MPMethod(task_name, directions_mp, input_dim=x_train.shape[-1])
        projection_mp = mp_object.mean_projection_method()

        # Clean data
        cleaned_x_train[train_indices] = np.dot(filt_x_train, projection_mp)  
        cleaned_x_dev[dev_indices] = np.dot(filt_x_dev, projection_mp) 
        cleaned_x_test[test_indices] = np.dot(filt_x_test, projection_mp) 

       # Free up memory
        del filt_x_train, filt_x_dev, filt_x_test, directions_mp, mp_object, projection_mp
        gc.collect()


    # Check if there is a row left with only zeros
    if np.any(np.all(cleaned_x_train == 0, axis=1)):
        print("Warning: cleaned_x_train contains a row with only zeros")
    if np.any(np.all(cleaned_x_dev == 0, axis=1)):
        print("Warning: cleaned_x_dev contains a row with only zeros")
    if np.any(np.all(cleaned_x_test == 0, axis=1)):
        print("Warning: cleaned_x_test contains a row with only zeros")

    # Save cleaned data
    if load_or_store_data:
        store_debiased_data(datapath, task_name, cleaned_x_train, cleaned_x_dev, cleaned_x_test)

    # return x_train_clean, x_dev_clean, x_test_clean
    return cleaned_x_train, cleaned_x_dev, cleaned_x_test





def get_INLP_cleaned_data_per_class(x_train, y_train, x_dev, y_dev, x_test, y_test, gender_train, gender_dev, gender_test, config, load_or_store_data=True):
    # task_name = "biasbios"   
    task_name = config['dataset'] + "_INLP_per_class"
    datapath = config['datapath']
    verbose = False

    if load_or_store_data and config["debias_load_stored"] is True:
        x_train_stored, x_dev_stored, x_test_stored = load_debiased_data(datapath, task_name)
        if x_train_stored is not None:
            return x_train_stored, x_dev_stored, x_test_stored

    # Get unique professions
    unique_professions = np.unique(y_train)

    # Create copies of the original data
    cleaned_x_train = np.zeros_like(x_train)
    cleaned_x_dev = np.zeros_like(x_dev)
    cleaned_x_test = np.zeros_like(x_test)

    # INLP params
    model = SGDClassifier
    loss = 'log_loss'
    warm_start = True
    early_stopping = False
    max_iter = 10000

    params = {'warm_start': warm_start, 'loss': loss, 'random_state': 0, 'early_stopping': early_stopping,
              'max_iter': max_iter}

    # Process each profession separately
    for prof_idx in tqdm.tqdm(unique_professions):
        # Filter data for current profession and get indices
        filt_x_train,  filt_gender_train, train_indices = get_prof_data(x_train, y_train, gender_train, prof_idx)
        filt_x_dev, filt_gender_dev, dev_indices = get_prof_data(x_dev, y_dev, gender_dev, prof_idx)
        filt_x_test,  filt_gender_test, test_indices = get_prof_data(x_test, y_test, gender_test, prof_idx)

        # print how many samples are in each class
        if verbose: print("train samples: ", len(filt_gender_train))

        filt_x_test_shuffled, filt_gender_train_shuffled = shuffle(filt_x_train, filt_gender_train, random_state=0, n_samples=len(filt_x_train))
        # Run INLP method
        inlp_object = INLPMethod(model, params)

        majority = Counter(filt_gender_dev).most_common(1)[0][1] / float(len(filt_gender_dev))
        if verbose: print('majority perc:', majority)

        # Main function for INLP
        output_p, all_projections, best_projection, accuracy_per_iteration = inlp_object.get_projection_matrix(50,
                                                                                                            filt_x_test_shuffled, 
                                                                                                            filt_gender_train_shuffled,
                                                                                                            filt_x_dev, filt_gender_dev,
                                                                                                            majority_acc=majority, verbose=verbose)
        # Clean data
        cleaned_x_train[train_indices] = np.dot(filt_x_train, output_p)  
        cleaned_x_dev[dev_indices] = np.dot(filt_x_dev, output_p) 
        cleaned_x_test[test_indices] = np.dot(filt_x_test, output_p) 

    # Check if there is a row left with only zeros
    if np.any(np.all(cleaned_x_train == 0, axis=1)):
        print("Warning: cleaned_x_train contains a row with only zeros")
    if np.any(np.all(cleaned_x_dev == 0, axis=1)):
        print("Warning: cleaned_x_dev contains a row with only zeros")
    if np.any(np.all(cleaned_x_test == 0, axis=1)):
        print("Warning: cleaned_x_test contains a row with only zeros")

    # Save cleaned data
    if load_or_store_data:
        store_debiased_data(datapath, task_name, cleaned_x_train, cleaned_x_dev, cleaned_x_test)

    return cleaned_x_train, cleaned_x_dev, cleaned_x_test


def get_debiase_method(config):
    method = config["debias_embs"]
    if method == "mp":
        if config["debias_per_class"] is True:
            print("MP - DEBIASING EMBEDDINGS BY CLASS")
            return get_MP_cleaned_data_per_class
        else: 
            print("MP - DEBIASING EMBEDDINGS - ALL CLASSES TOGETHER")
            return get_MP_cleaned_data_all
    elif method == "inlp":
        # return get_INLP_cleaned_data
        if config["debias_per_class"] is True:
            print("INLP - DEBIASING EMBEDDINGS BY CLASS")
            return get_INLP_cleaned_data_per_class
        else: 
            print("INLP - DEBIASING EMBEDDINGS - ALL CLASSES TOGETHER")
            return get_INLP_cleaned_data_all
    else:
        print(f"Unknown method: {method}")
        return None