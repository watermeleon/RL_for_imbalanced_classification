from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import pearsonr
import seaborn as sns


def get_prof_distri(all_data):
    """
    Calculate and return the normalized distribution of classes.
    - all_data (iterable): Collection of class labels.

    Returns:
    - dict: Normalized distribution of classes.
    """
    class_counts = Counter(all_data)
    total_samples = len(all_data)

    sorted_classes = sorted(class_counts)
    normalized_counts = [class_counts[i] / total_samples for i in sorted_classes]

    normalized_counts_dict = dict(zip(sorted_classes, normalized_counts))
    return normalized_counts_dict

# code inspired by https://github.com/tue-alga/debias-mean-projection/tree/main/notebooks
def load_dictionary(path):
    with open(path, "r", encoding = "utf-8") as f:
        lines = f.readlines()

    k2v, v2k = {}, {}
    for line in lines:
        k,v = line.strip().split("\t")
        v = int(v)
        k2v[k] = v
        v2k[v] = k

    return k2v, v2k

def count_profs_and_gender(data):
  # count number of individuals with a specific gender in every occupation
  # returns dictionary
    counter = defaultdict(Counter)
    for entry in data:
        gender, prof = entry["g"], entry["p"]
        counter[prof][gender] += 1.

    return counter

def get_prof2fem(all_data):
    counter = count_profs_and_gender(all_data)
    prof2fem = dict()

    for k, values in counter.items():
        prof2fem[k] = values['f']/(values['f'] + values['m'])

    return prof2fem


#code for evaluation metrics, modified from https://github.com/tue-alga/debias-mean-projection/tree/main/notebooks
def get_tpr(y_pred, y_true, i2p, test_genders):
    """
    Calculate the True Positive Rate (TPR) change for different genders.

    Parameters:
    y_pred (list): List of predicted values.
    y_true (list): List of true values.
    i2p (function): A function to map predicted values to some identity, if None, an identity function is used.
    test_genders (list): List of genders corresponding to each prediction.

    Returns:
    dict: A dictionary with the TPR change for each profession.
    """

    scores = defaultdict(Counter)
    prof_count_total = defaultdict(Counter)

    if i2p is None:
        i2p = np.arange(30)

    # if y_true is a tensor convert it to a list
    if hasattr(y_true, 'tolist'):
        # y_true = y_true.tolist()
        # first to cpu 
        y_true = y_true.cpu().tolist()


    for pred_prof, actual_prof, gender in zip(y_pred, y_true, test_genders):
        if int(actual_prof) == pred_prof:
            scores[i2p[actual_prof]][gender] += 1
        prof_count_total[i2p[actual_prof]][gender] += 1

    tpr_change = {}
    for profession, scores_dict in scores.items():
        total_males = prof_count_total[profession][0]
        total_females = prof_count_total[profession][1]

        correct_males = scores_dict[0]
        correct_females = scores_dict[1]

        tpr_change_value = 0.0000002
        if total_females != 0 and total_males != 0:
            tpr_male = correct_males / total_males
            tpr_female = correct_females / total_females
            tpr_change_value = tpr_female - tpr_male

        tpr_change[profession] = tpr_change_value
        
    return tpr_change

def get_tpr_per_gender(y_pred, y_true, i2p, test_genders):
    """
    Calculate the True Positive Rate (TPR) change for different genders.

    Parameters:
    y_pred (list): List of predicted values.
    y_true (list): List of true values.
    i2p (function): A function to map predicted values to some identity, if None, an identity function is used.
    test_genders (list): List of genders corresponding to each prediction.

    Returns:
    dict: A dictionary with the TPR change for each profession.
    """

    scores = defaultdict(Counter)
    prof_count_total = defaultdict(Counter)

    if i2p is None:
        i2p = np.arange(30)


    for pred_prof, actual_prof, gender in zip(y_pred, y_true, test_genders):
        if int(actual_prof) == pred_prof:
            scores[i2p[actual_prof]][gender] += 1
        prof_count_total[i2p[actual_prof]][gender] += 1

    tpr_change = {}
    for profession, scores_dict in scores.items():
        total_males = prof_count_total[profession][0]
        total_females = prof_count_total[profession][1]

        correct_males = scores_dict[0]
        correct_females = scores_dict[1]

        tpr_male = tpr_female = 0
        if total_females != 0:
            tpr_female = correct_females / total_females
        if  total_males != 0:
            tpr_male = correct_males / total_males

        tpr_change[profession] = {0: tpr_male, 1: tpr_female}
        
    return tpr_change

def similarity_vs_tpr(tpr_dict, title, measure, prof2fem, result_path,  title_name=None):
    """
    Plot the similarity vs TPR (True Positive Rate) for each profession.

    Args:
        tpr_dict (dict): A dictionary containing the TPR values for each profession.
        title (str): The title of the plot.
        measure (str): The measure used for similarity.
        prof2fem (dict): A dictionary mapping each profession to its similarity value.
        result_path (str): The path to save the resulting plot.

    Returns:
        tuple: A tuple containing the correlation and p-value of the plotted data.
    """
    professions = list(tpr_dict.keys())
    tpr_lst = [tpr_dict[p] for p in professions]
    sim_lst = [prof2fem[p] for p in professions]

    
    fontsize = 17
    sns.set_context("talk", font_scale=1.1)  # Set context to 'talk'
    sns.despine()

    plt.scatter(sim_lst, tpr_lst, s=30)
    sns.despine()

    plt.xlabel("frequency women", fontsize = fontsize + 2)
    plt.ylabel(r'$TPR_{gap}$', fontsize = fontsize)
    if title_name is not None:
        plt.title(title_name, fontsize = fontsize)
    for p in professions:
        x,y = prof2fem[p], tpr_dict[p]
        plt.annotate(p , (x,y), size = 10, color = "red")

    z = np.polyfit(sim_lst, tpr_lst, 1)
    p = np.poly1d(z)
    plt.plot(sim_lst,p(sim_lst),"b--")
    correlation, p_value =  pearsonr(sim_lst, tpr_lst)
    print("Correlation: {}; p-value: {}".format(correlation, p_value))

    # set y axis from -0.3 to 0.5
    plt.ylim(-0.3, 0.5)
    plt.tight_layout()
    plt.savefig(result_path + '_similarity_vs_tpr.pdf')
    plt.clf()

    return correlation, p_value



def calc_tpr_gap(tpr_gaps):
    """	 Calculate the root mean square of the TPR-GAPs. """	
    rms = np.sqrt(np.mean(np.square(np.array(list(tpr_gaps.values())))))
    return rms

def calc_tpr_gap_weighted(tpr_gaps, prof2perc):
    """	 Calculate the weighted average of the TPR-GAPs. """
    weighted_avg = sum([ abs(v)*prof2perc[k] for k,v in tpr_gaps.items()])
    return weighted_avg*100

def l2norm(matrix_1, matrix_2):
    """calculate Euclidean distance

    Args:
        matrix_1 (n*d np array): n is the number of instances, d is num of metric
        matrix_2 (n*d np array): same as matrix_1

    Returns:
        float: the row-wise Euclidean distance 
    """
    return np.power(np.sum(np.power(matrix_1-matrix_2, 2), axis=1), 0.5)

def DTO(fairness_metric, performacne_metric, utopia_fairness = None, utopia_performance = None):
    """calculate DTO for each condidate model

    Args:
        fairness_metric (List): fairness evaluation results (1-GAP)
        performacne_metric (List): performance evaluation results
    """
    
    fairness_metric, performacne_metric = np.array(fairness_metric), np.array(performacne_metric)
    # Best metric
    if (utopia_performance is None):
        utopia_performance = np.max(performacne_metric)
    if (utopia_fairness is None):
        utopia_fairness = np.max(fairness_metric)

    # Normalize
    performacne_metric = performacne_metric/utopia_performance
    fairness_metric = fairness_metric/utopia_fairness

    print("Utop Perf:", utopia_performance, "Utop Fair:", utopia_fairness)

    # Reshape and concatnate
    performacne_metric = performacne_metric.reshape(-1,1)
    fairness_metric = fairness_metric.reshape(-1,1)
    normalized_metric = np.concatenate([performacne_metric, fairness_metric], axis=1)

    # Calculate Euclidean distance
    return l2norm(normalized_metric, np.ones_like(normalized_metric))


def get_best_timestep(all_eval_metrics, selection_criterion = "DTO"):
    if selection_criterion == "DTO":
        # Extract fairness and performance metrics from all_eval_metrics
        fairness_metric = [metrics["fairness"] for metrics in all_eval_metrics.values()]
        performance_metric = [metrics["performance"] for metrics in all_eval_metrics.values()]
        
        # Calculate DTO for each candidate model
        dto_values = DTO(fairness_metric, performance_metric)
        
        # Find the best timestep
        best_timestep = list(all_eval_metrics.keys())[np.argmin(dto_values)]

        # best dev acc and tpr_gap_rms
        print("The best results are for:", all_eval_metrics[best_timestep])

        return best_timestep