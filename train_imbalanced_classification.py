import os

# Standard library imports
import json
import sys
from pathlib import Path

# Third party imports
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Local application/library specific imports
from imbalanced_classification.classifierenv_imb import ClassifierEnvGym_Imb
from imbalanced_classification.embedding_debias_methods.get_debiased_mp_embs import *
from imbalanced_classification.ppo.train_ppo import train_ppo, load_ppo_model
from imbalanced_classification.train_mab_agent import *
from utils import *
from utils.metrics_and_stat_functions import *
from utils.reward_scales import *


# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


subset_classes = [2, 11, 13, 18, 19, 21, 22, 26]
class_mapping = {cls: i for i, cls in enumerate(subset_classes)} # maps from old2new index


def args_update_config(config):
    for arg in sys.argv[1:]:
        print(arg)
        if arg.startswith('--'):
            arg_name = arg[2:]
            if "=" in arg_name:
                arg_name = arg_name.split("=")[0]
                arg_value = arg.split("=")[-1]
            else:
                arg_value = None
                if len(sys.argv) > sys.argv.index(arg) + 1 and not sys.argv[sys.argv.index(arg) + 1].startswith('--'):
                    arg_value = sys.argv[sys.argv.index(arg) + 1]
   
            print(f"{arg_name}: {arg_value}" + " " + str(type(arg_value)))
            arg_type = type(config[arg_name])
            if arg_type is bool:
                arg_value = arg_value.lower() == "true"
            else:
                arg_value = arg_type(arg_value)
            config[arg_name] = arg_value
    return config

    
if __name__ == '__main__':

    # Load the configuration file
    with open('config.json') as config_file:
        config = json.load(config_file)

    config = args_update_config(config)
    config["device"] = device    
    config["subset_classes"] = subset_classes
    config["class_mapping"] = class_mapping

    print(config)
    run_id = config["run_id"]

    # sleep 5 seconds times the run_id to avoid overwriting the same file
    import time
    n_secs = 5 * int(run_id)+0.1
    print(f"Sleeping for {n_secs} seconds")
    time.sleep(n_secs)

    all_train_types = ["ppo", "supervised", "dqn", "thompson", "linucb", "mp", "inlp"]
    train_type = config["train_type"]
    assert train_type in all_train_types, f"provided train_type is {train_type}, but must be one of {all_train_types}"
    assert config["dataset"] in ["biasbios", "emoji"], "dataset must be one of [biasbios, emoji]"

    wandb_name = "occup_class_Top8_LinUCB"
    if "wandb_name" in config and config["wandb_name"] is not False:
        wandb_name = config["wandb_name"]

    num_classes = ""
    if config["dataset"] == "biasbios":
        if config["use_most_common_classes"]:
            num_classes = "_8C"
        else:
            num_classes = "_28C"


    model_name = f"{config['exp_name']}_{config['dataset']}_{config['train_type']}_S{config['random_seed']}{num_classes}_R{config['reward_scale_type']}"
    config["model_name"] = model_name
    print("model_name", model_name)

    wandb_mode = "disabled" if  config["wandb_username"] == "None" else "online"
    wandb.init(project=wandb_name ,name=config['model_name'], entity=config["wandb_username"], mode="disabled")
    wandb.config.update(config)
    print(wandb.config)

    if config["use_wandbid_name"] is True:
        wandbid = wandb.run.id
        model_name = f"{model_name}_WBID{wandbid}"
        config["model_name"] = model_name
        print("model_name", model_name)
        wandb.config.update(config, allow_val_change=True)

    # set random_seeds
    random_seed = config["random_seed"]
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    num_workers = config["num_workers"]

    # define the paths
    repo_path = "./"


    # Set the user to distinguish between HPC and personal laptop
    dir_path = os.path.dirname(os.path.realpath(__file__)).split('/')
    user = "snelius"
    if "leonardo" in dir_path:
        print('Using Linux settings')
        user = "leonardo"
    config["user"] = user

    # load the data and init all hyperparameters
    if config["dataset"] == "biasbios":
        prof2fem, prof2perc, x_train, y_train, x_dev, y_dev, x_test, y_test, train_genders, test_genders, dev_genders = load_profession_data(config["datapath"], config)

        p2i, i2p = load_dictionary(config["datapath"] + 'profession2index.txt')
        prof2fem_indV2 = prof2fem
        prof2fem = {i2p[k]:v for k,v in prof2fem.items()}
        if config["use_most_common_classes"]:
            p2i, i2p, _, _ = remap_professions(p2i, i2p, prof2fem, prof2perc, subset_classes)
        prof2fem = {i2p[k]:v for k,v in prof2fem_indV2.items()}
        prof2perc = {i2p[k]:v for k,v in prof2perc.items()}
        print("prof2fem",prof2fem)
        print("prof2perc",prof2perc)
        config["i2p"] = i2p
        prof2fem_indx = {p2i[k]:v for k,v in prof2fem.items()}

    elif config["dataset"] == "emoji":
        prof2fem, _ , x_train, y_train, x_dev, y_dev, x_test, y_test, train_genders, test_genders, dev_genders = load_sentiment_data(config["datapath"], emoji_ratio=config["emoji_ratio"])
        prof2fem_indx = prof2fem 
        i2p = None


    # MP debiasing and sensitive information added explicitely.
    if config["gender_bool"] is True:
        print("Adding gender label to the input data")
        # add the gender value to the X data
        x_train = np.concatenate((x_train, train_genders.reshape(-1, 1)), axis=1)
        x_test = np.concatenate((x_test, test_genders.reshape(-1, 1)), axis=1)
        x_dev = np.concatenate((x_dev, dev_genders.reshape(-1, 1)), axis=1)
        
    if config["debias_embs"] != "none" and config["debiase_posthoc"] == False:
        print("Debiasing embeddings - before training")
        debias_function = get_debiase_method(config)
        x_train, x_dev, x_test = debias_function(x_train, y_train, x_dev, y_dev, x_test, y_test, train_genders, dev_genders, test_genders, config, load_or_store_data=True)

    input_size = x_train[0].shape[0]

    # training parameters
    model_name = config["model_name"]
    result_path = f"results/{model_name}/"
    Path(os.path.dirname(result_path)).mkdir(parents=True, exist_ok=True) # create parent folders of file
    config["result_path"] = result_path

    in_shape = [input_size, 194]
    if config["use_most_common_classes"]:
        num_classes = 8
    else:
        num_classes = len(np.unique(y_dev))

    lr = config["lr"]
    
    # convert data from numpy to torch tensors:
    def nump_to_torch(x):
        x =  torch.from_numpy(x)
        x.requires_grad = False
        return x.to(device)
    
    x_train = nump_to_torch(x_train.squeeze()).float()
    y_train = nump_to_torch(y_train).long()
    x_dev = nump_to_torch(x_dev.squeeze()).float()
    y_dev = nump_to_torch(y_dev).long()
    x_test = nump_to_torch(x_test.squeeze()).float()
    y_test = nump_to_torch(y_test).long()

    
    # assert reward_scale_type is one of the possible types
    reward_scale_type = config["reward_scale_type"]
    assert reward_scale_type in ["constant", "EO", "imb_ratio_plus", "imb_ratio_neg", "gender_and_prof"]

    train_env = ClassifierEnvGym_Imb(x_train, y_train, train_genders, prof2fem_indx, reward_scale_type=reward_scale_type)

    train_env.seed(random_seed)
    n_steps = int(len(x_train)*config["num_epoch"])

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
    
    def get_hidden_states(model, dataset_train, dataset_dev, dataset_test):
        model.eval()

        batch_size = 512
        dataloader_train = torch.utils.data.DataLoader(dataset_train, shuffle=False, batch_size=batch_size, num_workers=num_workers)
        dataloader_dev = torch.utils.data.DataLoader(dataset_dev, shuffle=False, batch_size=batch_size, num_workers=num_workers)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, shuffle=False, batch_size=batch_size, num_workers=num_workers)

        new_x_train = get_embeddings(model, dataloader_train)
        new_x_dev = get_embeddings(model, dataloader_dev)
        new_x_test = get_embeddings(model, dataloader_test)

        model.train()

        return new_x_train.cpu().detach().numpy(), new_x_dev.cpu().detach().numpy(), new_x_test.cpu().detach().numpy()

    if config["test_only"] is True:
        if config["train_type"] == "ppo":
            print("Loading PPO agent for testing")
            ppo_agent = load_ppo_model(train_env, input_size, num_classes, model_name, config)
            model = ppo_agent.policy_old.actor
        else:
            print("Loading MAB agent for testing")
            model = NeuralBandit2(num_classes, input_size, learning_rate=lr,n_steps=n_steps, config = config)
            model = model.model
            model.load_state_dict(torch.load(result_path + "_model.pt"))

    else:
        if config["train_type"] == "ppo":
            """ Train a PPO model """
            print("Training with PPO")
            dataset_val = FlexibleDataSet([x_dev, y_dev, dev_genders], device=device)
            imbalanced_penalty = create_reward_scale_list(prof2fem_indx, y_train, train_genders, config["reward_scale_type"])    

            # create dataloaders
            dataset_train = FlexibleDataSet([x_train, y_train, train_genders, imbalanced_penalty], device=device)
            dataloader_train = torch.utils.data.DataLoader(dataset_train, shuffle=True, batch_size=config["batch_size"], num_workers=num_workers)
            dataloader_val = torch.utils.data.DataLoader(dataset_val, shuffle=False, batch_size=512, num_workers=num_workers)
            
            ppo_agent, all_eval_metrics = train_ppo(train_env, dataloader_train, dataloader_val, input_size, num_classes, model_name, config, wandb)
            model = ppo_agent.policy_old.actor  

        elif config["train_type"] == "supervised" or config["train_type"]  in ["mp", "inlp"]:
            """ Supervised Learning"""
            print("Training with Supervised Learning")
            model = NeuralBandit2(num_classes, input_size, learning_rate=lr,n_steps=n_steps, config = config)

            # create dataloaders
            y_train_onehot =  F.one_hot(y_train, num_classes=num_classes).type(torch.FloatTensor)
            y_dev_onehot =  F.one_hot(y_dev, num_classes=num_classes).type(torch.FloatTensor)
            dataset_val = FlexibleDataSet([x_dev, y_dev_onehot, dev_genders], device=device)
            dataloader_val = torch.utils.data.DataLoader(dataset_val, shuffle=False, batch_size=512, num_workers=num_workers)
            
            imbalanced_penalty = create_reward_scale_list(prof2fem_indx, y_train, train_genders, config["reward_scale_type"])    


            # initialize dataloader training
            if config["reward_scale_type"] != "constant":
                dataset_train = FlexibleDataSet([x_train, y_train_onehot, imbalanced_penalty], device=device)
            else:
                dataset_train = FlexibleDataSet([x_train, y_train_onehot], device=device)
            dataloader_train = torch.utils.data.DataLoader(dataset_train, shuffle=True, batch_size=config["batch_size"], num_workers=num_workers)

            all_eval_metrics = model.train_supervised(dataloader_train, dataloader_val, config, wandb)
            model = model.model

            if config["debias_embs"] != "none" and config["debiase_posthoc"] == True:
                print("DEBIASING EMBEDDINGS - after supervised training")
                x_train, x_dev, x_test = get_hidden_states(model, FlexibleDataSet([x_train], device=device), FlexibleDataSet([x_dev], device=device), FlexibleDataSet([x_test], device=device))
                debias_function = get_debiase_method(config)
                y_train_np, y_dev_np, y_test_np = y_train.cpu().detach().numpy(), y_dev.cpu().detach().numpy(), y_test.cpu().detach().numpy()
                x_train, x_dev, x_test = debias_function(x_train, y_train_np, x_dev, y_dev_np, x_test, y_test_np, train_genders, dev_genders, test_genders, config, load_or_store_data=True)

                # make the x values as tensors
                x_train = nump_to_torch(x_train.squeeze()).float()
                x_dev = nump_to_torch(x_dev.squeeze()).float()
                x_test = nump_to_torch(x_test.squeeze()).float()




        elif config["train_type"] == "dqn" or config["train_type"] == "thompson" or config["train_type"] == "linucb":
            """ Deep Q-learning"""
            model = NeuralBandit2(num_classes, input_size, learning_rate=lr,n_steps=n_steps, config = config)
            dataset_val = FlexibleDataSet([x_dev, y_dev, dev_genders], device=device)
            dataloader_val = torch.utils.data.DataLoader(dataset_val, shuffle=False, batch_size=512, num_workers=num_workers)
            if config["train_type"] == "thompson": 
                all_eval_metrics = model.train_thompson_agent(train_env, n_steps, dataloader_val, config, wandb)
            elif config["train_type"] == "linucb":
                print("Training with LinUCB")
                model, all_eval_metrics = model.train_linucb_agent(train_env, n_steps, dataloader_val, config, wandb)
            else:
                print("Training with DQN")
                all_eval_metrics = model.train_dqn_agent(train_env, n_steps, dataloader_val, config, wandb)
                model = model.model

        # check if MP
        if config["train_type"] == "mp" or config["train_type"] == "inlp":
            print("Training with MP - logistic regression")
            from sklearn.linear_model import LogisticRegression

            classifier = LogisticRegression(warm_start = True, 
                                    penalty = 'l2',
                                    solver = "sag", 
                                    multi_class = 'multinomial', 
                                    fit_intercept = True,
                                    verbose = 0, 
                                    max_iter=10, 
                                    n_jobs = -1,          
                                    random_state = 1)
            
            classifier.fit(x_train.cpu().detach().numpy(), y_train.cpu().detach().numpy())

            model = lambda x: (torch.Tensor(classifier.predict(x.cpu().detach().numpy())), None)
            all_eval_metrics = {1:{"eval_dto_dist": 0.0}, 2:{"eval_dto_dist": 0.0}}


    # all dto_eval_vals 
    all_dto_eval_vals = [item["eval_dto_dist"] for item in all_eval_metrics.values()]
    best_eval_dto = np.min(all_dto_eval_vals)

    #predict on test set
    print("test res:", model(x_test[0].unsqueeze(0)))
    y_pred = evaluate_on_test_set(model, x_test, y_test, device=device)

    # if y_pred is a tensor, convert to list
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy().tolist()
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.cpu().numpy().tolist()

    #write predictions to file
    pred_y_path = repo_path + result_path +  'pred_1'
    Path(os.path.dirname(pred_y_path)).mkdir(parents=True, exist_ok=True) # create parent folders of file
    np.savetxt(pred_y_path, y_pred)


    tpr_gaps = get_tpr(y_pred, y_test, i2p, test_genders)
    tpr_gap_rms_test = calc_tpr_gap(tpr_gaps)
    print("RMS TPR Gap:"+str(tpr_gap_rms_test))

    print("classification report")
    print(classification_report(y_pred, y_test))
    test_accuracy = accuracy_score(y_test, y_pred)
    f1_score_macro = f1_score(y_test, y_pred, average='macro')

    print("correlation")
    correlation, p_value = similarity_vs_tpr(tpr_gaps, "TPR scores BL 1", "TPR", prof2fem, result_path)

    test_metrics = {"correlation": correlation, "p_value": p_value, "test_accuracy": test_accuracy, "tpr_gap_rms_test": tpr_gap_rms_test, 
                    "f1_score_macro": f1_score_macro, "best_eval_dto": best_eval_dto}
    
    # join the dict tpr_gaps to test metrics
    if "store_tpr_wandb" in config and  config["store_tpr_wandb"] is True:
        test_metrics.update(tpr_gaps)
    
    print("test_metrics", test_metrics)
    wandb.log(test_metrics)
    wandb.finish()
