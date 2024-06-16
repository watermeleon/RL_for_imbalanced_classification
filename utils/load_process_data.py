import numpy as np
import pickle
from .metrics_and_stat_functions import get_prof_distri

def remap_professions(p2i, i2p, prof2fem, prof2perc, subset_classes):

    p2i_new = {}
    i2p_new = {}
    prof2fem_new = {}
    prof2perc_new = {}

    for new_ind, old_ind in enumerate(subset_classes):
        profession = i2p[old_ind]
        p2i_new[profession] = new_ind
        i2p_new[new_ind] = i2p[old_ind]

        # actually based on prof so k:v stays the same, but removes the professions that are not in subset_classes
        # prof2fem_new[profession] = prof2fem[profession]
        # prof2perc_new[new_ind] = prof2perc[old_ind]

    print("New i2p:", i2p_new)
    return p2i_new, i2p_new, prof2fem_new, prof2perc_new



def get_subset_by_gender(X, Y, genders, gender_percentage_dict={}, config=None):
    """
    This function returns a subset of elements (X, Y, genders) that matches the desired gender ratio for each profession
    specified in gender_percentage_dict. It adjusts the subset by subsampling for one gender to match the desired percentage.

    Parameters:
    - X: NumPy array, contains features or attributes.
    - Y: NumPy array, contains the class labels (professions).
    - genders: NumPy array, contains gender attribute.
    - gender_percentage_dict: Dictionary, keys are professions and values are the desired percentage of females.

    Returns:
    - subset_X: Subset of X with adjusted gender ratios.
    - subset_Y: Subset of Y with adjusted gender ratios.
    - subset_genders: Subset of genders with adjusted gender ratios.
    """
    subset_X, subset_Y, subset_genders = [], [], []
    subset_classes=config["subset_classes"]
    class_mapping = config["class_mapping"]

    # Unique professions
    unique_professions = np.unique(Y)


    for profession in unique_professions:
        if profession not in subset_classes:
            continue
        profession_mask = Y == profession
        X_profession = X[profession_mask]
        Y_profession = Y[profession_mask]
        gender_profession = genders[profession_mask]

        if profession in gender_percentage_dict:
            desired_percentage_female = gender_percentage_dict[profession]

            # Calculate the number of females needed to achieve the desired percentage
            total_profession = len(X_profession)
            num_females_needed = int(total_profession * desired_percentage_female)
            female_mask = gender_profession == 1
            male_mask = np.logical_not(female_mask)
            
            actual_num_females = np.sum(female_mask)
            actual_num_males = np.sum(male_mask)

            if actual_num_females > num_females_needed:
                # More females than needed, subsample females
                females_to_select = num_females_needed
                males_to_select = actual_num_males  # Keep all males
            else:
                # Not enough females, calculate how many males to keep to meet the desired percentage
                females_to_select = actual_num_females  # Keep all females
                # The target number of males to match the desired percentage of females
                males_to_select = min(actual_num_males, int((actual_num_females / desired_percentage_female) - actual_num_females))

            # Selecting subsets
            females_selected = np.where(female_mask)[0][:females_to_select]
            males_selected = np.where(male_mask)[0][:males_to_select]
            
            selected_indices = np.concatenate((females_selected, males_selected))
            
            subset_X.extend(X_profession[selected_indices])
            subset_Y.extend(Y_profession[selected_indices])
            subset_genders.extend(gender_profession[selected_indices])
            
        else:
            subset_X.extend(X_profession)
            subset_Y.extend(Y_profession)
            subset_genders.extend(gender_profession)
    
    subset_Y_new_idx = np.array([class_mapping[y] for y in subset_Y])
    return np.array(subset_X), np.array(subset_Y_new_idx), np.array(subset_genders)


def get_prof2fem_v2(Y, genders):
    """
    This function calculates the percentage of females for each profession.

    Parameters:
    - Y: NumPy array, contains the class labels (professions).
    - genders: NumPy array, contains gender attribute ('Male', 'Female', etc.).

    Returns:
    - prof2fem: Dictionary, keys are professions and values are the percentage of females in that profession.
    """
    prof2fem = {}
    unique_professions = np.unique(Y)

    for profession in unique_professions:
        # Filter data for the current profession
        profession_mask = Y == profession
        gender_profession = genders[profession_mask]

        # Count females in this profession - female is index 1, male index 0
        female_count = np.sum(gender_profession == 1)

        # Calculate percentage
        percentage_female = (female_count / len(gender_profession)) 
        prof2fem[profession] = percentage_female

    return prof2fem

def loaddata(datapath, config):
    # the folder for the input data X :
    data_folder = config["datafolder"] + "/"
 
    x_train = np.load(datapath + data_folder + 'train_input_ids.npy', allow_pickle =True)
    x_dev =np.load(datapath + data_folder + 'dev_input_ids.npy', allow_pickle =True)
    x_test =np.load(datapath + data_folder + 'test_input_ids.npy',  allow_pickle =True)

    with open (datapath + data_folder + 'train_labels', 'rb') as fp:
        y_train = pickle.load(fp)
    with open (datapath + data_folder + 'dev_labels', 'rb') as fp:
        y_dev = pickle.load(fp)
    with open (datapath + data_folder + 'test_labels', 'rb') as fp:
        y_test = pickle.load(fp)

    with open (datapath + data_folder + 'train_gender_list', 'rb') as fp:
        train_genders = np.array(pickle.load(fp))
    with open (datapath + data_folder + 'test_gender_list', 'rb') as fp:
        test_genders = np.array(pickle.load(fp))
    with open (datapath + data_folder + 'dev_gender_list', 'rb') as fp:
        dev_genders = np.array(pickle.load(fp))

    if config["use_most_common_classes"]:
        if config["skew_data"] is True:
            gender_percentage_dict = {22:0.9, 2:0.2}
        else:
            gender_percentage_dict = {} # empty means no skewing
        x_train, y_train, train_genders = get_subset_by_gender(x_train, y_train, train_genders, gender_percentage_dict=gender_percentage_dict, config=config)
        x_dev, y_dev, dev_genders =  get_subset_by_gender(x_dev, y_dev, dev_genders, config=config)
        x_test, y_test, test_genders = get_subset_by_gender(x_test, y_test, test_genders, config=config)



    # print lengths of splits:
    print("train length, X:", x_train.shape, ", y:", y_train.shape, ", Gender:", len(train_genders))
    print("dev length, X:", x_dev.shape, ", y:", y_dev.shape, ", Gender:", len(dev_genders))
    print("test length, X:", x_test.shape, ", y:", y_test.shape, ", Gender:", len(test_genders))
    
    prof2perc = get_prof_distri(y_train)
    prof2fem = get_prof2fem_v2(y_train, train_genders)
    print("prof2fem",prof2fem)

    return prof2fem, prof2perc, x_train, y_train, x_dev, y_dev, x_test, y_test, train_genders, test_genders, dev_genders
