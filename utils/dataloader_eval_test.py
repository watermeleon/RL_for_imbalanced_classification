import numpy as np

import torch

# add sys one path up to import utils
import sys
sys.path.append("..")

from utils.metrics_and_stat_functions import get_tpr, calc_tpr_gap
from tqdm import tqdm


class FlexibleDataSet(torch.utils.data.Dataset):
    def __init__(self, data_sets, device="cpu"):
        """
        Initialize the dataset with an arbitrary number of tensors or lists.
        Each argument represents a different variable.
        """
        super(FlexibleDataSet, self).__init__()

        # Function to get the first dimension size
        def get_first_dim_size(data):
            if isinstance(data, list):
                return len(data)
            elif isinstance(data, np.ndarray) or torch.is_tensor(data):
                return data.shape[0]
            else:
                raise TypeError("Unsupported data type. Only lists, numpy arrays, and PyTorch tensors are allowed.")

        # Check that all inputs have the same first dimension size
        first_dim_size = get_first_dim_size(data_sets[0])

        if not all(get_first_dim_size(tensor) == first_dim_size for tensor in data_sets):
            raise ValueError("All inputs must have the same size in the first dimension")

        # self.tensors = [torch.tensor(tensor, device=device) if isinstance(tensor, list) else tensor for tensor in data_sets]
        self.tensors = [torch.as_tensor(tensor, device=device) if not isinstance(tensor, torch.Tensor) else tensor.to(device) for tensor in data_sets]

    def __len__(self):
        """
        Return the size of the dataset (defined by the first dimension of the tensors).
        """
        return self.tensors[0].shape[0]

    def __getitem__(self, index):
        """
        Return the items at the given index from each tensor.
        """
        return tuple(tensor[index] for tensor in self.tensors)
    

def evaluate_on_validation_set_batchwise(model, dataloader_val, supervised = False, val_presplit=False):
    # model.eval()


    def supervised_operator(labels):
        """converts one hot encoded labels to categorical labels"""
        # return np.argmax(labels.numpy(), 1)
        return labels.argmax(dim=1).cpu().numpy()
    
    def rl_operator(labels):
        """ tiny operator to improve speed :)"""
        return labels.cpu().numpy()
    

    if supervised is True:
        label_operator = supervised_operator
    else:
        label_operator = rl_operator

    with torch.no_grad():
        if val_presplit is True:
            all_inputs, all_labels, all_gender = dataloader_val  
        else:
            inputs_list = []
            labels_list = []
            gender_list = []
            output_list = []

            # Accumulate all the inputs and labels first
            for i, data in enumerate(tqdm(dataloader_val), 0):
                inputs, labels, gender = data
                inputs_list.append(inputs)
                labels_list.append(labels)
                gender_list.append(gender)

                outputs, _  = model(inputs)
                output_list.append(torch.tensor(outputs))

            # Stack all inputs and labels
            all_inputs = torch.cat(inputs_list, dim=0)
            all_labels = torch.cat(labels_list, dim=0)
            all_gender = torch.cat(gender_list, dim=0)
            all_outputs = torch.cat(output_list, dim=0)

        # Pass all inputs through the model at once
        # all_outputs = model(all_inputs)

        # Process all outputs and calculate accuracy
        all_labels_cat = label_operator(all_labels)
        # all_pred_cat = all_outputs.detach().numpy()
        all_pred_cat = all_outputs.detach().cpu().numpy()

        acc = (all_labels_cat == all_pred_cat).astype(float)

        tpr_gap = get_tpr(all_pred_cat, all_labels_cat.tolist(), None, all_gender.tolist())
        tpr_gap_rms= calc_tpr_gap(tpr_gap)
        tpr_gap_rms = torch.tensor(tpr_gap_rms, dtype=torch.float32).item()

        eval_acc = np.mean(acc)
    return eval_acc, tpr_gap, tpr_gap_rms

def evaluate_on_validation_set(model, dataloader_val, supervised = False, val_presplit=False):
    print("Testing on validation set")
    model.eval()

    def supervised_operator(labels):
        """converts one hot encoded labels to categorical labels"""
        # return np.argmax(labels.numpy(), 1)
        return labels.argmax(dim=1).cpu().numpy()
    
    def rl_operator(labels):
        """ tiny operator to improve speed :)"""
        return labels.cpu().numpy()
    
    if supervised is True:
        label_operator = supervised_operator
    else:
        label_operator = rl_operator

    with torch.no_grad():
        if val_presplit is True:
            all_inputs, all_labels, all_gender = dataloader_val  
        else:
            inputs_list = []
            labels_list = []
            gender_list = []

            # Accumulate all the inputs and labels first
            for data in dataloader_val:
                inputs, labels, gender = data
                inputs_list.append(inputs)
                labels_list.append(labels)
                gender_list.append(gender)

            # Stack all inputs and labels
            all_inputs = torch.cat(inputs_list, dim=0)
            all_labels = torch.cat(labels_list, dim=0)
            all_gender = torch.cat(gender_list, dim=0)

        # Pass all inputs through the model at once
        all_outputs = model(all_inputs)

        # Process all outputs and calculate accuracy
        all_labels_cat = label_operator(all_labels)
        # all_pred_cat = np.argmax(all_outputs.detach().cpu().numpy(), axis=1)
        all_pred_cat = all_outputs.argmax(dim=1).detach().cpu().numpy()
        acc = (all_labels_cat == all_pred_cat).astype(float)

        tpr_gap = get_tpr(all_pred_cat, all_labels_cat.tolist(), None, all_gender.tolist())
        tpr_gap_rms= calc_tpr_gap(tpr_gap)
        tpr_gap_rms = torch.tensor(tpr_gap_rms, dtype=torch.float32).item()

        eval_acc = np.mean(acc)

    model.train()
    return eval_acc, tpr_gap, tpr_gap_rms


def evaluate_on_test_set(model, x_test, y_test, device):
    dataset_test = FlexibleDataSet([x_test, y_test], device=device)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, shuffle=False, batch_size=512, num_workers=0)

    # check if model has a function to evaluate
    if hasattr(model, 'eval'):
        model.eval()
    y_pred = []
    with torch.no_grad():
        for data in dataloader_test:
            inputs, _ = data
            outputs = model(inputs)
            # check if outputs is a tuple
            if isinstance(outputs, tuple):
                " for the cmab models, TODO: change this later" 
                all_pred_cat = outputs[0].detach().cpu().numpy()
            else:
                all_pred_cat = np.argmax(outputs.detach().cpu().numpy(), 1)
            y_pred.extend(all_pred_cat)

    return y_pred