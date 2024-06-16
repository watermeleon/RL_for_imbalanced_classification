import numpy as np
import pickle
import torch


def reward_scale_matrix_EO(prof2fem):
    """ the function is : 1/2 * 1/(P(gender|profession)) """
    num_professions = len(prof2fem)
    num_genders = 2
    reward_scale = np.ones((num_professions, num_genders))
    for i in range(num_professions):
        perc_female = prof2fem[i]
        for j in range(num_genders):
            # this will ensure scale = 1 for the other minority 
            if j == 1:
                # female
                gender_perc = perc_female
            else:
                # male
                gender_perc = 1 - perc_female

            scale = 1/2 * 1/(gender_perc)
            reward_scale[i, j] = scale

    return reward_scale

def reward_scale_matrix_imb_ratio_plus(prof2fem):
    """ set the reward scale to 1 for the minority, and imb ratio for the majority"""
    num_professions = len(prof2fem)
    num_genders = 2
    reward_scale = np.ones((num_professions, num_genders))
    for i in range(num_professions):
        perc_female = prof2fem[i]
        for j in range(num_genders):
            # this will ensure scale = 1 for the other minority 
            majority_percentage = minority_percentage = 1
            if j == 1 and perc_female > 0.5:
                # female in a female dominated profession
                minority_percentage = 1 - perc_female
                majority_percentage = perc_female
            elif j == 0 and perc_female < 0.5:
                # male in male dominated profession
                minority_percentage = perc_female
                majority_percentage = 1 - perc_female

            scale =  minority_percentage / majority_percentage
            reward_scale[i, j] = scale

    return reward_scale



def reward_scale_matrix_imb_ratio_neg(prof2fem):
    """ set the reward scale to 1 for the majority, and 1/(imb ratio) for the minority"""
    num_professions = len(prof2fem)
    num_genders = 2
    reward_scale = np.ones((num_professions, num_genders))
    for i in range(num_professions):
        perc_female = prof2fem[i]
        for j in range(num_genders):
            # this will ensure scale = 1 for the other majority 
            majority_percentage = minority_percentage = 1
            if j == 1 and perc_female < 0.5:
                # female in a male dominated profession
                minority_percentage = perc_female
                majority_percentage = 1 - perc_female
            elif j == 0 and perc_female > 0.5:
                # male in a female dominated profession
                minority_percentage = 1 - perc_female
                majority_percentage = perc_female

            scale = majority_percentage / minority_percentage
            reward_scale[i, j] = scale

    return reward_scale


def reward_scale_matrix_gender_and_prof(prof2fem):
    with open('./data/prof_distr_dict.pkl', "rb") as f:
        prof_distr_dict = pickle.load(f)
    prof_distr_matrix = [prof_distr_dict[i] for i in range(len(prof_distr_dict))]
    prof_distr_matrix = (1/ np.array(prof_distr_matrix)) 
    prof_distr_matrix = prof_distr_matrix / np.min(prof_distr_matrix)
    
    # round to 3 decimals
    prof_distr_matrix = np.round(prof_distr_matrix, 3)

    rs_matrix_neg = reward_scale_matrix_imb_ratio_neg(prof2fem)

    for i in range(len(rs_matrix_neg)):
        rs_matrix_neg[i] *= prof_distr_matrix[i]

    return rs_matrix_neg


def get_reward_scale_matrix(reward_scale_type, prof2fem):
    """   get the reward scale matrix"""
    print("using reward scales V4")
    if reward_scale_type == "constant":
        rs_matrix = np.ones((len(prof2fem), 2))
    elif reward_scale_type == "EO":
        rs_matrix = reward_scale_matrix_EO(prof2fem)
    elif reward_scale_type == "imb_ratio_plus":
        rs_matrix = reward_scale_matrix_imb_ratio_plus(prof2fem)
    elif reward_scale_type == "imb_ratio_neg":
        rs_matrix = reward_scale_matrix_imb_ratio_neg(prof2fem)
    elif reward_scale_type == "gender_and_prof":
        rs_matrix = reward_scale_matrix_gender_and_prof(prof2fem)
    else:
        raise ValueError("reward_scale_type not recognized")
    
    rs_matrix = np.round(rs_matrix, 3)
    return rs_matrix

def reward_scale_per_datapoint(rs_matrix, prof_labels, gender_labels):
    """ set the majority to 1.0, and the minority to 1/(imb ratio)."""
    reward_scale_list = []

    for i in range(len(prof_labels)):
        profession = prof_labels[i]
        gender = gender_labels[i]
        scale = rs_matrix[profession, gender]
        reward_scale_list.append(scale)

    return torch.tensor(reward_scale_list)


def create_reward_scale_list(prof2fem, prof_labels, gender_labels, reward_scale_type):
    """ get the reward scales using the technique defined in config 
        first get the reward scale matrix, then use it to get the reward scale for each datapoint."""
    
    rs_matrix = get_reward_scale_matrix(reward_scale_type, prof2fem)
    reward_scale_list = reward_scale_per_datapoint(rs_matrix, prof_labels, gender_labels)

    return reward_scale_list


