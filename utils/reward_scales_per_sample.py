import numpy as np
import pickle


def reward_scale_imb_ratio_EO(prof2fem, prof_labels, gender_labels):
    """ set the minority to 1.0, and the majority to the imb ratio."""
    reward_scale_list = []

    for i in range(len(prof_labels)):
        profession = prof_labels[i]
        gender = gender_labels[i]
        perc_fem = prof2fem[profession.item()]
        
        if gender == 1:
            # female
            gender_perc = perc_fem
        else:
            # male
            gender_perc = 1 - perc_fem

        # the function is : 1/2 * 1/(P(gender|profession))
        scale = 1/2 * 1/(gender_perc)
        reward_scale_list.append(scale)
    
    return reward_scale_list

def reward_scale_imb_ratio_plus(prof2fem, prof_labels, gender_labels):
    """ set the minority to 1.0, and the majority to the imb ratio."""
    reward_scale_list = []

    for i in range(len(prof_labels)):
        profession = prof_labels[i]
        gender = gender_labels[i]
        perc_fem = prof2fem[profession.item()]
        
        # this will ensure scale = 1 for the other minority 
        majority_percentage = minority_percentage = 1
        if gender == 1 and perc_fem > 0.5:
            # female in a female dominated profession
            minority_percentage = 1 - perc_fem
            majority_percentage = perc_fem
        elif gender == 0 and perc_fem < 0.5:
            # male in male dominated profession
            minority_percentage = perc_fem
            majority_percentage = 1 - perc_fem

        scale =  minority_percentage / majority_percentage
        reward_scale_list.append(scale)

    return reward_scale_list



def reward_scale_matrix_imb_ratio_plus(self, prof2fem):
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

def reward_scale_imb_ratio_neg(prof2fem, prof_labels, gender_labels):
    """ set the majority to 1.0, and the minority to 1/(imb ratio)."""
    reward_scale_list = []

    for i in range(len(prof_labels)):
        profession = prof_labels[i]
        gender = gender_labels[i]
        perc_fem = prof2fem[profession.item()]
        
        # this will ensure scale = 1 for the other majority 
        majority_percentage = minority_percentage = 1
        if gender == 1 and perc_fem < 0.5:
            # female in a male dominated profession
            minority_percentage =  perc_fem
            majority_percentage = 1 - perc_fem
        elif gender == 0 and perc_fem > 0.5:
            # male in female dominated profession
            minority_percentage = 1 - perc_fem
            majority_percentage = perc_fem

        scale =  majority_percentage / minority_percentage
        reward_scale_list.append(scale)

    return reward_scale_list


def reward_scale_matrix_imb_ratio_neg(self, prof2fem):
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



def reward_scale_gender_and_prof(prof2fem, prof_labels, gender_labels, config):
    scale_prof_imb = config["scale_reward_prof_imb"] 
    if scale_prof_imb is True:
        # scale each profession by its specific 1/prof_perc
        with open('./data/prof_distr_dict.pkl', "rb") as f:
            prof_distr_dict = pickle.load(f)
        prof_distr_matrix = [prof_distr_dict[i] for i in range(len(prof_distr_dict))]
        prof_distr_matrix = (1/ np.array(prof_distr_matrix)) 
        prof_distr_matrix = prof_distr_matrix / np.min(prof_distr_matrix)
        
        # round to 3 decimals
        prof_distr_matrix = np.round(prof_distr_matrix, 3)

    reward_scale_list =  reward_scale_imb_ratio_neg(prof2fem, prof_labels, gender_labels)

    for i in range(len(prof_labels)):
        reward_scale_list[i] *= prof_distr_matrix[prof_labels[i]]

    return reward_scale_list
def reward_scale_constant(prof2fem, prof_labels, gender_labels):
    """ set reward scale to 1.0 for all data points."""
    reward_scale_list = [1.0] * len(prof_labels)
    
    return reward_scale_list


def create_reward_scale_list(prof2fem, prof_labels, gender_labels, config):
    """ get the reward scales using the technique defined in config """
    reward_scale_list = []
    print("using reward scales V4")

    reward_scale_type = config["reward_scale_type"]
    if reward_scale_type == "constant":
        reward_scale_list = reward_scale_constant(prof2fem, prof_labels, gender_labels) 
    elif reward_scale_type == "EO":
        reward_scale_list = reward_scale_imb_ratio_EO(prof2fem, prof_labels, gender_labels)
    elif reward_scale_type == "imb_ratio_plus":
        reward_scale_list = reward_scale_imb_ratio_plus(prof2fem, prof_labels, gender_labels)
    elif reward_scale_type == "imb_ratio_neg":
        reward_scale_list = reward_scale_imb_ratio_neg(prof2fem, prof_labels, gender_labels)
    elif reward_scale_type == "gender_and_prof":
        reward_scale_list = reward_scale_gender_and_prof(prof2fem, prof_labels, gender_labels, config)
    else:
        raise ValueError("reward_scale_type not recognized")


    reward_scale_list = np.round(reward_scale_list, 3)

    return reward_scale_list





######### The previous Reward Scale functions ####################

def create_gender_minority_list_V3(prof2fem, prof_labels, gender_labels, config):
    """ set the majority to 1.0, and the minority increased."""
    reward_scale_list = []
    print("using reward scales V3")

    scale_prof_imb = config["scale_reward_prof_imb"] 
    if scale_prof_imb is True:
        # scale each profession by its specific 1/imb_perc
        with open('./data/prof_distr_dict.pkl', "rb") as f:
            prof_distr_dict = pickle.load(f)
        prof_distr_matrix = [prof_distr_dict[i] for i in range(len(prof_distr_dict))]
        prof_distr_matrix = (1/ np.array(prof_distr_matrix)) 
        prof_distr_matrix = prof_distr_matrix / np.min(prof_distr_matrix)
        # scale it so that the minimum is 1 and the maximum is 10
        # prof_distr_matrix = 1 + (prof_distr_matrix - np.min(prof_distr_matrix)) * 9 / (np.max(prof_distr_matrix) - np.min(prof_distr_matrix))
        
        # round to 3 decimals
        prof_distr_matrix = np.round(prof_distr_matrix, 3)


    for i in range(len(prof_labels)):
        profession = prof_labels[i]
        gender = gender_labels[i]
        perc_fem = prof2fem[profession.item()]
        
        if gender == 1 and perc_fem < 0.5:
            # female in a male dominated profession
            minority_percentage = 1 - perc_fem
            scale = minority_percentage / (perc_fem)
            # scale *= 2    # This is an ablation to seee how the models handle overshooting
        elif gender == 0 and perc_fem > 0.5:
            # male in female dominated profession
            minority_percentage = perc_fem
            scale = minority_percentage / (1 - perc_fem)
            # scale *= 2    # This is an ablation to seee how the models handle overshooting
        else:
            scale = 1

        if scale_prof_imb is True:
            scale *= prof_distr_matrix[profession]

        reward_scale_list.append(scale)

    reward_scale_list = np.round(reward_scale_list, 3)
    # print("reward scales V3 are here:", reward_scale_list)
    return reward_scale_list

def create_gender_minority_list(prof2fem, prof_labels, gender_labels, config):
    """
    Creates a list indicating whether the gender in a given data point is in the minority for the corresponding profession.

    Parameters:
    - prof2fem (dict): A dictionary mapping professions to the percentage of females in each profession.
    - prof_labels (list): A list of profession labels for each data point.
    - gender_labels (list): A list of gender labels (0 for male, 1 for female) corresponding to each data point.

    Returns:
    - reward_scale_list (list): A list indicating whether the gender in each data point is in the minority for the corresponding profession.
      - For minority gender in the profession: 1 
      - For majority gender in the profession: The percentage of the minority in that profession.

    """
    reward_scale_list = []

    for i in range(len(prof_labels)):
        profession = prof_labels[i]
        gender = gender_labels[i]

        perc_fem = prof2fem[profession.item()]
        
        if gender == 1 and perc_fem > 0.5:
            # female in a female dominated profession
            if config["rs_relative_perc"] is True:
                minority_percentage = 1 - perc_fem
                scale = minority_percentage / (perc_fem)
                reward_scale_list.append(scale)
            else:
                reward_scale_list.append(1- perc_fem)
        elif gender == 0 and perc_fem < 0.5:
            # male in male dominated profession
            if config["rs_relative_perc"] is True:
                minority_percentage = perc_fem
                scale = minority_percentage / (1 - perc_fem)
                reward_scale_list.append(scale)
            else:
                reward_scale_list.append(perc_fem)
        else:
            reward_scale_list.append(1)

    return reward_scale_list
