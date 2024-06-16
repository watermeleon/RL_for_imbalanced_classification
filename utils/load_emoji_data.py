import numpy as np
import os
from collections import Counter


def adjust_class_distribution(x_train, y_train, protected_attributes_train, sentiment, target_perc_minority):
    """
    Adjusts the class distribution for a given sentiment to target a specific percentage of minority samples.
    """
    indices = np.where(y_train == sentiment)[0]
    minority_indices = indices[protected_attributes_train[indices] == 1]
    majority_indices = indices[protected_attributes_train[indices] == 0]

    current_perc_minority = len(minority_indices) / len(indices)
    if target_perc_minority is not None:
        if target_perc_minority < current_perc_minority:
            # Need to downsample the minority class
            np.random.shuffle(minority_indices)
            if target_perc_minority > 0.5:
                # Upsample majority to decrease minority percentage
                num_majority_needed = int(len(minority_indices) * (1 - target_perc_minority) / target_perc_minority) - len(majority_indices)
                majority_indices = np.concatenate([majority_indices, np.random.choice(majority_indices, size=num_majority_needed, replace=True)])
            else:
                # Downsample minority to decrease its percentage
                num_minority_to_keep = int(len(majority_indices) * target_perc_minority / (1 - target_perc_minority))
                minority_indices = minority_indices[:num_minority_to_keep]
        else:
            # Need to downsample the majority class or upsample minority
            np.random.shuffle(majority_indices)
            if target_perc_minority < 0.5:
                # Upsample minority to increase minority percentage
                num_minority_needed = int(len(majority_indices) * target_perc_minority / (1 - target_perc_minority)) - len(minority_indices)
                minority_indices = np.concatenate([minority_indices, np.random.choice(minority_indices, size=num_minority_needed, replace=True)])
            else:
                # Downsample majority to increase minority percentage
                num_majority_to_keep = int(len(minority_indices) * (1 - target_perc_minority) / target_perc_minority)
                majority_indices = majority_indices[:num_majority_to_keep]

    new_indices = np.concatenate([minority_indices, majority_indices])
    np.random.shuffle(new_indices)  # Shuffle to mix minority and majority indices
    return x_train[new_indices], y_train[new_indices], protected_attributes_train[new_indices]

def calculate_and_print_combination_frequencies(y_train, protected_attributes_train, y_test, protected_attributes_test, y_dev, protected_attributes_dev):
    def calculate_combination_frequencies(y, protected_attributes, set_name):
        # Combine y and protected_attribute into tuples for each training example
        y_protected_attribute_combinations = list(zip(y, protected_attributes))
        
        # Count the occurrences of each combination
        combination_counts = Counter(y_protected_attribute_combinations)
        
        # Calculate total count
        total_count = sum(combination_counts.values())
        
        # Print each combination, its count, and its percentage of the total
        print(f"{set_name}:")
        for (y, protected_attribute), count in combination_counts.items():
            protected_attribute_name = 'AAE' if protected_attribute == 1 else 'SAE'
            emotion = 'HAPPY' if y == 1 else 'SAD'
            percentage = (count / total_count) * 100
            print(f'Combination: protected_attribute = {protected_attribute_name}, Emotion = {emotion}, Count = {count}, Percentage = {percentage:.2f}%')
        print("--------------------")
        

    calculate_combination_frequencies(y_train, protected_attributes_train, "Training set")
    calculate_combination_frequencies(y_test, protected_attributes_test, "Testing set")
    calculate_combination_frequencies(y_dev, protected_attributes_dev, "Development set")



def load_sentiment_data(emoji_datapath, pos_perc_minority=0.8, neg_perc_minority=0.2, emoji_ratio=0.0):
    """protected_attribute_label: pos:AAE = 1, neg:SAE = 0 """
    emoji_datapath = os.path.join(emoji_datapath, 'deepmoji/')

    # emoji_ratio = A stereotyping level of symmetric 0.8 means 80:20 black:white for positive and 20:80 black:white for negative
    if emoji_ratio > 0:
        pos_perc_minority= emoji_ratio
        neg_perc_minority= 1 - emoji_ratio        


    # Helper function to load data from a specific path
    def load_data_from_files(folder_path):
        data_files = ['pos_neg.npy', 'pos_pos.npy', 'neg_neg.npy', 'neg_pos.npy']
        sentiment_labels = []
        protected_attribute_labels = []
        data_vectors = []

        for file_name in data_files:
            file_path = os.path.join(folder_path, file_name)
            data = np.load(file_path)

            # Extract sentiment and protected_attribute from filename
            sentiment, protected_attribute = file_name.split('.')[0].split('_')
            sentiment_label = 1 if sentiment == 'pos' else 0  # 1 for positive, 0 for negative sentiment
            protected_attribute_label = 1 if protected_attribute == 'pos' else 0  # 1 for AAE, 0 for SAE

            data_vectors.extend(data)
            sentiment_labels.extend([sentiment_label] * len(data))
            protected_attribute_labels.extend([protected_attribute_label] * len(data))

        return np.array(data_vectors), np.array(sentiment_labels), np.array(protected_attribute_labels)
    
    # Load training, development, and testing data
    x_train, y_train, protected_attributes_train = load_data_from_files(os.path.join(emoji_datapath, 'train'))
    x_dev, y_dev, protected_attributes_dev = load_data_from_files(os.path.join(emoji_datapath, 'dev'))
    x_test, y_test, protected_attributes_test = load_data_from_files(os.path.join(emoji_datapath, 'test'))

    if pos_perc_minority is not None or neg_perc_minority is not None:
        def adjust_data_distribution(x_labels, y_labels, protected_attributes_labels, pos_perc_minority, neg_perc_minority):
            # Adjust training data distribution if necessary
            x_labels_pos, y_labels_pos, protected_attributes_labels_pos = adjust_class_distribution(x_labels, y_labels, protected_attributes_labels, 1, pos_perc_minority)
            x_labels_neg, y_labels_neg, protected_attributes_labels_neg = adjust_class_distribution(x_labels, y_labels, protected_attributes_labels, 0, neg_perc_minority)
            
            # Combine adjusted positive and negative datasets back into a single training dataset
            x_labels = np.concatenate([x_labels_pos, x_labels_neg])
            y_labels = np.concatenate([y_labels_pos, y_labels_neg])
            protected_attributes_labels = np.concatenate([protected_attributes_labels_pos, protected_attributes_labels_neg])

            return x_labels, y_labels, protected_attributes_labels

        x_train, y_train, protected_attributes_train = adjust_data_distribution(x_train, y_train, protected_attributes_train, pos_perc_minority, neg_perc_minority)
        # x_dev, y_dev, protected_attributes_dev = adjust_data_distribution(x_dev, y_dev, protected_attributes_dev, pos_perc_minority, neg_perc_minority)
        # x_test, y_test, protected_attributes_test = adjust_data_distribution(x_test, y_test, protected_attributes_test, pos_perc_minority, neg_perc_minority)
    

    # Calculate sentiment2aa
    sentiment2aa = {}
    for sentiment in [0, 1]:  # 0 for negative, 1 for positive
        indices = np.where(y_train == sentiment)[0]
        if len(indices) > 0:
            protected_attribute_counts = np.sum(protected_attributes_train[indices])
            sentiment2aa[sentiment] = round(protected_attribute_counts / len(indices), 3)
        else:
            sentiment2aa[sentiment] = None  # Handle case with no instances


    calculate_and_print_combination_frequencies(y_train, protected_attributes_train, y_test, protected_attributes_test, y_dev, protected_attributes_dev)

    # sentiment2aa now contains the percentage of minority for each sentiment in the training set
    return sentiment2aa, None, x_train, y_train, x_dev, y_dev, x_test, y_test, protected_attributes_train, protected_attributes_test, protected_attributes_dev


