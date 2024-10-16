# example run: python train.py -l training_results/main/0-RO -c undersampled_csvs/main/0-RO --RO

import sys
import os

sys.path.append("./models")
sys.path.append("..")

custom_module_path = '/data/private/BreizhCrops' 
if custom_module_path not in sys.path:
    sys.path.insert(0, custom_module_path)

import argparse
from get_results import get_results
import breizhcrops
from breizhcrops.models import LSTM, TempCNN, MSResNet, TransformerModel, InceptionTime, StarRNN, OmniScaleCNN, PETransformerModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
import torch
import torch.nn as nn
import torch.hub
import pandas as pd
import geopandas as gpd
import sklearn.metrics
from sklearn.metrics import confusion_matrix
from imblearn.metrics import geometric_mean_score
from fvcore.nn import FlopCountAnalysis, parameter_count
import time
from sklearn.metrics import precision_score, recall_score
import torch.nn.functional as F
import random
import numpy as np
from collections import Counter
from scipy.stats import gaussian_kde
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit, train_test_split
from torch.utils.data import Subset
import re


class EarlyStopping:
    def __init__(self, patience=20, metric='gmean'):
        self.patience = patience
        self.metric = metric
        self.best_metric = None
        self.epochs_no_improve = 0
        self.stop_training = False
        self.first_improvement_observed = False  # New flag to track the first improvement

    def check(self, metrics):
        current_metric = metrics[self.metric]
        if self.best_metric is None:
            self.best_metric = current_metric
        elif current_metric > self.best_metric:
            self.best_metric = current_metric
            self.epochs_no_improve = 0
            self.first_improvement_observed = True  # Set flag to True on first improvement
        else:
            if self.first_improvement_observed:  # Only count patience after first improvement
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    self.stop_training = True

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def load_max_weights(csv_file_path, new_id_mapping):
    max_weights = {}
    df = pd.read_csv(csv_file_path)
    for index, row in df.iterrows():
        original_class_id = int(row['classid'])
        # Apply new_id_mapping to ensure consistency with dataset's class IDs
        if original_class_id in new_id_mapping:
            mapped_class_id = new_id_mapping[original_class_id]
            max_weights[mapped_class_id] = row['max_weight']
    print("Max weights after applying new_id_mapping:", max_weights)
    return max_weights

def rename_logdir(base_logdir, best_value):
    new_logdir = f"{base_logdir}{best_value}"
    if not os.path.exists(new_logdir):
        os.rename(base_logdir, new_logdir)
    return new_logdir

def calculate_class_weights(dataset, device):
    total_distribution = Counter()
    
    # Iterate through the dataset to get class distributions
    for i in range(len(dataset)):
        sample = dataset[i]
        if len(sample) == 4:
            _, y, _, _ = sample
        elif len(sample) == 3:
            _, y, _ = sample
        else:
            raise ValueError("Unexpected sample length")
        
        class_distribution = Counter([y.item()])
        total_distribution.update(class_distribution)
    
    total_samples = sum(total_distribution.values())
    num_classes = len(total_distribution)
    weights = torch.zeros(num_classes, device=device)
    
    for class_id, count in total_distribution.items():
        class_weight = total_samples / (num_classes * count)
        weights[class_id] = class_weight
        
    print(weights)
    return weights


def validate_dataset(dataset, num_samples=5):
    print(f"Validating dataset with {num_samples} samples...")
    for i in range(num_samples):
        try:
            X, y, area, row_id = dataset[i]
            print(f"Sample {i}: Row ID = {row_id}, Area = {area}")
            # Optionally, add more checks or print statements here
        except ValueError as e:
            print(f"Validation failed: {e}")
            break  # Stop validation if an error is encountered
    print("Validation completed.")


def find_last_checkpoint(epoch_save_dir):
    """Find the last saved checkpoint based on the file naming convention."""
    checkpoints = [f for f in os.listdir(epoch_save_dir) if f.startswith('model_state_epoch_') and f.endswith('.pth')]
    
    if not checkpoints:
        return None, 0  # No checkpoint found, start from epoch 0

    # Extract epoch numbers from filenames and find the latest
    epochs = [int(re.findall(r'model_state_epoch_(\d+).pth', ckpt)[0]) for ckpt in checkpoints]
    last_epoch = max(epochs)
    
    checkpoint_path = os.path.join(epoch_save_dir, f'model_state_epoch_{last_epoch}.pth')
    return checkpoint_path, last_epoch

def calculate_modes_and_thresholds(csv_paths, gdf, percent=10):
    # Combine data from all specified CSV files
    combined_df = pd.DataFrame()
    for csv_path in csv_paths:
        temp_df = pd.read_csv(csv_path)
        combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

        
    # Debug: Print combined_df
    print("Combined DataFrame after loading CSVs and applying new_id_mapping:")
    print(combined_df.head())
    
    # Merge with GeoDataFrame
    # Ensure 'id' is integer for accurate merging
    combined_df['id'] = combined_df['id'].astype(int)
    gdf['id'] = gdf['id'].astype(int)
    merged_df = pd.merge(combined_df, gdf[['id', 'area']], on='id')

    # Verbose output for sanity checks
    if merged_df['area'].isnull().any():
        print("Warning: Missing area information for some parcels.")
    if (merged_df['classid'] == -1).any():
        print("Warning: Unmapped class IDs detected.")

    # Debug: Print merged_df
    print("Merged DataFrame:")
    print(merged_df.head())

    # Calculate mode for each class using KDE
    class_modes = {}
    bottom_percent_thresholds = {}
    for class_id in merged_df['classid'].unique():
        if class_id == -1:
            continue  # Skip unmapped class IDs
        class_areas = merged_df.loc[merged_df['classid'] == class_id, 'area'].values
        if class_areas.size > 0:
            kde = gaussian_kde(class_areas)
            linspace = np.linspace(min(class_areas), max(class_areas), 1000)
            mode_area = linspace[np.argmax(kde(linspace))]
            class_modes[class_id] = mode_area

            # Calculate bottom percent threshold
            sorted_areas = np.sort(class_areas)
            threshold_index = int(len(sorted_areas) * percent / 100)
            bottom_percent_thresholds[class_id] = sorted_areas[threshold_index]

    # Verbose output for final results
    print(f"Class Modes: {class_modes}")
    print(f"Bottom Percent Thresholds: {bottom_percent_thresholds}")

    return class_modes, bottom_percent_thresholds


def inspect_sample_structure(dataset):
    sample = dataset[0]
    print(f"Sample structure: {len(sample)} elements")
    for i, element in enumerate(sample):
        if isinstance(element, torch.Tensor):
            print(f"Element {i}: Type {type(element)}, Shape: {element.shape}")
        elif hasattr(element, '__len__'):
            print(f"Element {i}: Type {type(element)}, Length: {len(element)}")
        else:
            print(f"Element {i}: Type {type(element)}, Value: {element}")


class doubleObjectiveWeightedCrossEntropy(nn.Module):
    def __init__(self, class_modes, bottom_percent_thresholds, max_weights_per_class, class_weights):
        super(doubleObjectiveWeightedCrossEntropy, self).__init__()
        self.class_modes = class_modes
        self.bottom_percent_thresholds = bottom_percent_thresholds
        self.max_weights_per_class = max_weights_per_class
        self.class_weights = class_weights
        


    def forward(self, predictions, targets, area):
        standard_loss = F.cross_entropy(predictions, targets, weight=self.class_weights, reduction='none')
        size_weights = torch.ones_like(targets).float()
        
        for i in range(len(area)):
            class_id = targets[i].item()

            mode_area = self.class_modes[class_id]
            max_weight = self.max_weights_per_class.get(class_id) 
    
            if area[i] < mode_area:  # Below the mode
                if area[i] < self.bottom_percent_thresholds[class_id]:  # Smallest 10%
                    size_weights[i] = max_weight
                else:
                    # Linearly interpolate weight based on area's position between the mode and the 10% threshold
                    threshold_area = self.bottom_percent_thresholds[class_id]
                    
                    size_weights[i] = max_weight - (area[i] - threshold_area) / (mode_area - threshold_area) * (max_weight - 1)
            # Areas above the mode automatically have a weight of 1

        weighted_loss = standard_loss * size_weights
        return weighted_loss.mean()


def create_subset(traindatasets, subset_ratio=0.3, random_state=42):
    y = [traindatasets[i][1].item() for i in range(len(traindatasets))]  # Extract labels
    _, subset_idx = train_test_split(np.arange(len(traindatasets)), test_size=subset_ratio, stratify=y, random_state=random_state)
    subset = Subset(traindatasets, subset_idx)
    return subset


def create_balanced_subset(traindatasets, random_state=42):
    def remove_duplicates(dataset):
        seen_ids = set()
        unique_samples = []
        for sample in dataset:
            _, y, row_id = sample[:3]  # Extracting row_id assuming it's the third element
            if row_id not in seen_ids:
                seen_ids.add(row_id)
                unique_samples.append(sample)
        return unique_samples
    
    # Set the random seed
    random.seed(random_state)

    # Remove duplicates
    unique_samples = remove_duplicates(traindatasets)
    
    # Sanity check before sampling
    class_counts_before = Counter([sample[1].item() for sample in unique_samples])
    print("Class distribution before sampling:", class_counts_before)

    # Determine the minimum number of samples for any class
    min_samples_per_class = min(class_counts_before.values())

    # Separate samples by class
    class_samples = {class_id: [] for class_id in class_counts_before}
    for sample in unique_samples:
        label = sample[1].item()
        class_samples[label].append(sample)
    
    # Undersample to achieve balance
    balanced_samples = []
    for class_id, samples in class_samples.items():
        sampled_samples = random.sample(samples, min_samples_per_class)
        balanced_samples.extend(sampled_samples)
    
    # Shuffle the balanced dataset
    random.shuffle(balanced_samples)
    
    # Sanity check after sampling
    balanced_class_counts = Counter([sample[1].item() for sample in balanced_samples])
    print("Class distribution after sampling:", balanced_class_counts)
    
    return balanced_samples


def cross_validate(args, traindatasets, datasets, param_grid, param_name, meta, subset_ratio=0.3):
    if args.RO:
        traindatasets = create_balanced_subset(traindatasets)
    else:
        # Create a subset of the data for cross-validation
        traindatasets = create_subset(traindatasets, subset_ratio)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_param = None
    best_score = -np.inf
    cv_results = []

    param_tqdm = tqdm(param_grid, desc='Hyperparameter Tuning')
    
    
    for param in param_tqdm:
        fold_scores = []
        
        y = [traindatasets[i][1].item() for i in range(len(traindatasets))]  # Extract labels for stratification
        
        skf_tqdm = tqdm(skf.split(traindatasets, y), total=skf.get_n_splits(), desc='K-Folds')
        for fold_idx, (train_idx, val_idx) in enumerate(skf_tqdm):
            train_subset = torch.utils.data.Subset(traindatasets, train_idx)
            val_subset = torch.utils.data.Subset(traindatasets, val_idx)
            
            # Sanity check: print class proportions in train and validation sets
            y_train = [traindatasets[i][1].item() for i in train_idx]
            y_val = [traindatasets[i][1].item() for i in val_idx]
            train_class_counts = np.bincount(y_train, minlength=len(datasets[0].classname))
            val_class_counts = np.bincount(y_val, minlength=len(datasets[0].classname))
            train_class_proportions = train_class_counts / len(y_train)
            val_class_proportions = val_class_counts / len(y_val)
            print(f"Fold {fold_idx+1} for param {param}:")
            print(f"  Train class proportions: {train_class_proportions}")
            print(f"  Val class proportions: {val_class_proportions}")
            
            traindataloader = DataLoader(train_subset, batch_size=args.batchsize, shuffle=True, num_workers=args.workers)
            valdataloader = DataLoader(val_subset, batch_size=args.batchsize, shuffle=False, num_workers=args.workers)
            
            sample = train_subset[0]
            if len(sample) == 4:
                ndims = sample[0].shape[-1]
                sequencelength = sample[0].shape[0]
            elif len(sample) == 3:
                ndims = sample[0].shape[-1]
                sequencelength = sample[0].shape[0]
            else:
                raise ValueError("Unexpected sample length")
                
            class_names = datasets[0].classname  # If datasets is a list of dataset objects
            class_names = [name.replace(" ", "_") for name in class_names]


            meta.update({
                "class_names": class_names,
            })

            
            print(f"Meta information: {meta}")
            
            model = get_model(args.model, meta['ndims'], meta['num_classes'],meta['sequencelength'], args.device, **args.hyperparameter)
            optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            
            if args.loss == 'focal_loss':
                class_weights = calculate_class_weights(train_subset, args.device)
                criterion = torch.hub.load(
                    'adeelh/pytorch-multi-class-focal-loss',
                    model='FocalLoss',
                    alpha=class_weights,
                    gamma=param,
                    reduction='mean',
                    force_reload=False
                )
            elif args.loss == 'doubleObjectiveWeightedCrossEntropy':
                class_weights = calculate_class_weights(train_subset, args.device)
                max_weights_per_class = {class_id: param for class_id in range(meta['num_classes'])}
                print(f" max_weights_per_class for param {param}: {max_weights_per_class}")
                criterion = doubleObjectiveWeightedCrossEntropy(
                    class_modes=meta['class_modes'],
                    bottom_percent_thresholds=meta['bottom_percent_thresholds'],
                    max_weights_per_class=max_weights_per_class,
                    class_weights=class_weights
                )

            
            early_stopper = EarlyStopping(patience=7, metric='gmean')
            
            epoch_tqdm = tqdm(range(args.epochs), desc='Epochs', leave=False)
            for epoch in epoch_tqdm:
                train_loss = train_epoch(model, optimizer, criterion, traindataloader, args.device, args, meta)
                val_loss, y_true, y_pred, *_ = test_epoch(model, criterion, valdataloader, args.device)
                scores = metrics(y_true.cpu(), y_pred.cpu(), meta['num_classes'], meta['class_names'])
                early_stopper.check(scores)
                if early_stopper.stop_training:
                    break
                # Update the progress bar description with current scores
                epoch_tqdm.set_description(f"Epoch {epoch+1}/{args.epochs}, G-Mean: {scores['gmean']:.4f}")
                print(f"Fold {fold_idx+1}, Param {param}, Epoch {epoch+1} finished")
            
            fold_scores.append(scores)
        
        avg_scores = {metric: np.mean([score[metric] for score in fold_scores]) for metric in fold_scores[0]}
        cv_results.append((param, avg_scores))
        
        if avg_scores['gmean'] > best_score:
            best_score = avg_scores['gmean']
            best_param = param

        # Update the parameter progress bar description with the best score
        param_tqdm.set_description(f"Best Param: {best_param}, Best G-Mean: {best_score:.4f}")

    return best_param, cv_results



def train(args):

    traindataloader, validdataloader, traindatasets, meta, datasets, class_modes, bottom_percent_thresholds, max_weights_per_class, new_id_mapping = get_dataloader(args.datapath, args.csv_folder, args.batchsize, args.workers, args.preload_ram, args.level)
    
        # Inspect the sample structure
    inspect_sample_structure(traindatasets)
    
    num_batches_per_epoch = len(traindataloader)  # Number of batches in one epoch
    
    
    class_names = datasets[0].classname  # If datasets is a list of dataset objects
    class_names = [name.replace(" ", "_") for name in class_names]
    

    num_classes = meta["num_classes"]
    print(num_classes)
    ndims = meta["ndims"]
    sequencelength = meta["sequencelength"]

    device = torch.device(args.device)
    model = get_model(args.model, ndims, num_classes, sequencelength, device, **args.hyperparameter)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model.modelname += f"_learning-rate={args.learning_rate}_weight-decay={args.weight_decay}"
    print(f"Initialized {model.modelname}")
    


    logdir = args.logdir
    os.makedirs(logdir, exist_ok=True)
    epoch_save_dir = os.path.join(logdir, "epoch_saves")
    os.makedirs(epoch_save_dir, exist_ok=True)
    print(f"Logging results to {logdir}")

    
    meta.update({
        "class_modes": class_modes,
        "bottom_percent_thresholds": bottom_percent_thresholds,
        "max_weights_per_class": max_weights_per_class
    })
    
    # Check if we need to resume from a checkpoint
    start_epoch = 0
    if args.checkpoint:
        checkpoint_path, start_epoch = find_last_checkpoint(epoch_save_dir)
        if checkpoint_path:
            print(f"Resuming from checkpoint: {checkpoint_path} (epoch {start_epoch})")
            model.load_state_dict(torch.load(checkpoint_path))
        else:
            print("No checkpoint found. Starting training from scratch.")

    # Print the length of datasets
    print(f"Length of concatenated train datasets: {len(traindatasets)}")
    
    if args.loss == 'focal_loss':
        if args.focal_loss_gamma is None:
            best_gamma, cv_results = cross_validate(args, traindatasets,datasets, [0.5, 1, 1.5, 2, 2.5, 3], 'gamma',meta)
            args.focal_loss_gamma = best_gamma
            logdir = rename_logdir(logdir, best_gamma)
            save_cv_results(cv_results, logdir, 'focal_loss_gamma')
        class_weights = calculate_class_weights(traindatasets, device)
        criterion = torch.hub.load(
            'adeelh/pytorch-multi-class-focal-loss',
            model='FocalLoss',
            alpha=class_weights,
            gamma=args.focal_loss_gamma,
            reduction='mean',
            force_reload=False
        )
    elif args.loss == "weightedCrossEntropy":
        class_weights = calculate_class_weights(traindatasets, device) 
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
    elif args.loss == 'doubleObjectiveWeightedCrossEntropy':
        if args.max_small_parcel_weight is None:
            best_weight, cv_results = cross_validate(args, traindatasets,datasets, [1.25, 1.5, 2, 3, 4], 'max_small_parcel_weight',meta)
            args.max_small_parcel_weight = best_weight
            logdir = rename_logdir(logdir, best_weight)
            save_cv_results(cv_results, logdir, 'max_small_parcel_weight')
            max_weights_per_class = {class_id: args.max_small_parcel_weight for class_id in new_id_mapping.values() if class_id != -1}
            print(f"Max weights per class: {max_weights_per_class}")
        class_weights = calculate_class_weights(traindatasets, device)
        criterion = doubleObjectiveWeightedCrossEntropy(
            class_modes=class_modes,
            bottom_percent_thresholds=bottom_percent_thresholds,
            max_weights_per_class=max_weights_per_class,
            class_weights=class_weights
        )
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        
    epoch_save_dir = os.path.join(logdir, "epoch_saves")
    os.makedirs(epoch_save_dir, exist_ok=True)
    
    
    # Calculating FLOPs and parameter count
    dummy_input = torch.randn(1, sequencelength, ndims).to(device)
    flops = FlopCountAnalysis(model, dummy_input)
    flops_per_batch = flops.total()
    params = parameter_count(model)['']
    
    cumulative_flops = 0
    
    early_stopper = EarlyStopping(patience=20, metric='gmean')
    

    log = list()

        
    for epoch in range(start_epoch+1, args.epochs):
        start_time = time.time()
        
        train_loss = train_epoch(model, optimizer, criterion, traindataloader, device, args,meta)
        val_loss, y_true, y_pred, *_ = test_epoch(model, criterion, validdataloader, device)
        
        end_time = time.time()
        epoch_runtime = end_time - start_time
        
        print(f"Expected num_classes: {meta['num_classes']}, Actual class_names: {len(class_names)}")
        print(f"Class names: {class_names}")

        scores = metrics(y_true.cpu(), y_pred.cpu(), num_classes=meta['num_classes'], class_names=class_names)
        
        early_stopper.check(scores)
        if early_stopper.stop_training:
            print(f"Stopping early at epoch {epoch} due to no improvement in G-mean.")
            break
            
        scores_msg = ", ".join([f"{k}={v:.2f}" for (k, v) in scores.items()])
        val_loss = val_loss.cpu().detach().numpy()[0]
        train_loss = train_loss.cpu().detach().numpy()[0]
        print(f"epoch {epoch}: trainloss {train_loss:.2f}, valloss {val_loss:.2f} " + scores_msg)
        
        flops_per_epoch = flops_per_batch * num_batches_per_epoch
        cumulative_flops += flops_per_epoch

        scores["epoch"] = epoch
        scores["trainloss"] = train_loss
        scores["valloss"] = val_loss
        scores["FLOPs_per_epoch"] = flops_per_epoch
        scores["cumulative_FLOPs"] = cumulative_flops
        scores["runtime_per_epoch"] = epoch_runtime
        scores["Params"] = params
        log.append(scores)

        log_df = pd.DataFrame(log).set_index("epoch")
        log_df.to_csv(os.path.join(logdir, "trainlog.csv"))
        torch.save(model.state_dict(), os.path.join(epoch_save_dir, f"model_state_epoch_{epoch}.pth"))

        
    # Save the model's state
    torch.save(model.state_dict(), os.path.join(logdir, "final_model_state.pth"))

    # Load the primary GeoJSON file containing  parcel areas
    print("Loading geodataframe...")
    geojson_file_path = '/data/private/BreizhCrops/data/all_parcels.geojson'
    geojson_gdf = gpd.read_file(geojson_file_path)
    
    excluded_classes_from_area_feature = args.exclude_area_feature_for_classes



    get_results(logdir, args.csv_folder, geojson_gdf, args.datapath, num_classes, ndims,excluded_classes_from_area_feature, preload_ram=args.preload_ram, level=args.level)


def save_cv_results(cv_results, logdir, param_name):
    records = []
    for param, scores in cv_results:
        record = {"param": param}
        record.update(scores)
        records.append(record)
    cv_results_df = pd.DataFrame(records)
    cv_results_df.to_csv(os.path.join(logdir, f'cv_results_{param_name}.csv'), index=False)



def get_dataloader(datapath, csv_folder, batchsize, workers, preload_ram=False, level="L1C"):

    print(f"Batch size received in get_dataloader: {batchsize}")
    print(f"Setting up datasets in {os.path.abspath(datapath)}, level {level}")
    datapath = os.path.abspath(datapath)
    print(args.datapath)
    
    
    frh01_csv = os.path.join(csv_folder, 'frh01.csv')
    frh02_csv = os.path.join(csv_folder, 'frh02.csv')
    frh03_csv = os.path.join(csv_folder, 'frh03.csv')
    
    if args.RO:
        

        frh01_val_csv = os.path.join(csv_folder, 'frh01_val.csv')
        frh02_val_csv = os.path.join(csv_folder, 'frh02_val.csv')
        frh03_val_csv = os.path.join(csv_folder, 'frh03_val.csv')

        frh01 = breizhcrops.BreizhCrops(region="frh01", root=datapath, preload_ram=preload_ram, level=level, csv_file_name=frh01_csv, exclude_bands=args.exclude_bands)
        frh02 = breizhcrops.BreizhCrops(region="frh02", root=datapath, preload_ram=preload_ram, level=level, csv_file_name=frh02_csv, exclude_bands=args.exclude_bands)
        frh03 = breizhcrops.BreizhCrops(region="frh03", root=datapath, preload_ram=preload_ram, level=level, csv_file_name=frh03_csv, exclude_bands=args.exclude_bands)
        
        frh01_val = breizhcrops.BreizhCrops(region="frh01", root=datapath, preload_ram=preload_ram, level=level, csv_file_name=frh01_val_csv, exclude_bands=args.exclude_bands)
        frh02_val = breizhcrops.BreizhCrops(region="frh02", root=datapath, preload_ram=preload_ram, level=level, csv_file_name=frh02_val_csv, exclude_bands=args.exclude_bands)
        frh03_val = breizhcrops.BreizhCrops(region="frh03", root=datapath, preload_ram=preload_ram, level=level, csv_file_name=frh03_val_csv, exclude_bands=args.exclude_bands)
        
    else:

        frh01 = breizhcrops.BreizhCrops(region="frh01", root=datapath, preload_ram=preload_ram, level=level, csv_file_name=frh01_csv, exclude_bands=args.exclude_bands)
        frh02 = breizhcrops.BreizhCrops(region="frh02", root=datapath, preload_ram=preload_ram, level=level, csv_file_name=frh02_csv, exclude_bands=args.exclude_bands)
        frh03 = breizhcrops.BreizhCrops(region="frh03", root=datapath, preload_ram=preload_ram, level=level, csv_file_name=frh03_csv, exclude_bands=args.exclude_bands)
    
    if args.OHIT:
        OHIT_csv = os.path.join(csv_folder, 'ohit.csv')
        global_mapping = frh01.mapping.copy()

        synth = breizhcrops.BreizhCrops(region="OHIT", root=datapath,
                                        preload_ram=preload_ram, level=level, csv_file_name=OHIT_csv, exclude_bands=args.exclude_bands)
        synth.update_class_ids(global_mapping=global_mapping)


    if args.RO:

        train_dataset = torch.utils.data.ConcatDataset([frh01, frh02, frh03])
        val_dataset = torch.utils.data.ConcatDataset([frh01_val, frh02_val, frh03_val])

        
        '''
        # SANITY CHECK / DEBUG
        # Debug: Print the lengths of the datasets and the concatenated datasets
        print(f"Length of train_dataset: {len(train_dataset)}")
        print(f"Length of val_dataset: {len(val_dataset)}")

        # Verify the full dataset class distribution before splitting
        def count_classes(dataset):
            class_counts = Counter()
            for i in range(len(dataset)):
                try:
                    class_id = dataset[i][1].item()
                    class_counts[class_id] += 1
                except IndexError:
                    continue  # Ignore out-of-bounds indices
            return class_counts
        
        full_class_counts = count_classes(torch.utils.data.ConcatDataset([frh01, frh02, frh03,frh01_val, frh02_val, frh03_val]))
        print("Full dataset class counts before split:", full_class_counts)

        # Calculate class distributions considering duplicates
        train_class_counts = count_classes(train_dataset)
        val_class_counts = count_classes(val_dataset)

        print("Training set class counts after split:", train_class_counts)
        print("Validation set class counts after split:", val_class_counts)
        '''
        
        traindataloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.workers)
        validdataloader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.workers)

        # Set traindatasets to the concatenated training dataset
        traindatasets = train_dataset
        
    else:
        # Combine datasets to form the full training dataset
        traindatasets = torch.utils.data.ConcatDataset([frh01, frh02, frh03])
        
        # Stratified 80/20 split for validation
        y = [traindatasets[i][1].item() for i in range(len(traindatasets))]  # Extract labels
        '''
        # Sanity check before split
        original_class_counts = Counter(y)
        print("Original class counts:", original_class_counts)
        '''
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(splitter.split(np.zeros(len(y)), y))
        
        '''
        # Sanity check after split
        y_train = [y[i] for i in train_idx]
        y_val = [y[i] for i in val_idx]
        train_class_counts = Counter(y_train)
        val_class_counts = Counter(y_val)
        print("Training set class counts after split:", train_class_counts)
        print("Validation set class counts after split:", val_class_counts)
        '''
        
        train_dataset = torch.utils.data.Subset(traindatasets, train_idx)
        val_dataset = torch.utils.data.Subset(traindatasets, val_idx)
        
        traindataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=workers)
        validdataloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=workers)
        
        # Set traindatasets to the concatenated training dataset
        traindatasets = train_dataset    


    
    class_modes = None
    bottom_percent_thresholds = None
    max_weights_per_class = None
    
    meta = dict(
        ndims=13 if level == "L1C" else 10,
        num_classes = max(len(frh01.classes), len(frh02.classes), len(frh03.classes)),
        sequencelength=45

    )
    
        # Print sizes of individual datasets
    print(f"Size of frh01: {len(frh01)}")
    print(f"Size of frh02: {len(frh02)}")
    print(f"Size of frh03: {len(frh03)}")

    
    num_classes=meta['num_classes']
    
    new_id_mapping = frh01.new_id_mapping

    if args.use_area_feature or args.loss == 'doubleObjectiveWeightedCrossEntropy':
        
        print("Loading geodataframe...")
        gdf = gpd.read_file('/data/private/BreizhCrops/data/all_parcels.geojson')
        if args.OHIT:
            # Load the GeoDataFrame containing areas for the synthetic samples
            synth_gdf = gpd.read_file('/path/to/synthetic_areas.geojson')

            # Concatenate the two GeoDataFrames
            gdf = pd.concat([gdf, synth_gdf], ignore_index=True)
            
        area = gdf.set_index('id')['area']
        
        if args.loss == 'doubleObjectiveWeightedCrossEntropy':
            if args.RO:
                for dataset in (frh01, frh02, frh03, frh01_val, frh02_val, frh03_val):  # Include validation datasets when using --RO
                    dataset.include_area = True
                if args.OHIT:
                    synth.include_area = True

                print("Updated class IDs mapping to maintain original order:\n", new_id_mapping)

                class_modes, bottom_percent_thresholds = calculate_modes_and_thresholds(
                    [frh01_csv, frh02_csv, frh03_csv, frh01_val_csv, frh02_val_csv, frh03_val_csv], gdf, percent=10)  # Include validation CSVs
                print(f"Sample class modes: {dict(list(class_modes.items())[:5])}")
                print(f"Sample bottom percent thresholds: {dict(list(bottom_percent_thresholds.items())[:5])}")

                for dataset in (frh01, frh02, frh03, frh01_val, frh02_val, frh03_val):  # Set area for validation datasets
                    dataset.area = area

                if args.max_weights_csv:
                    max_weights_per_class = load_max_weights(args.max_weights_csv, new_id_mapping)
                else:
                    max_weights_per_class = {class_id: args.max_small_parcel_weight for class_id in new_id_mapping.values() if class_id != -1}

                print("Final max_weights_per_class:", max_weights_per_class)  # Debug print
                for dataset in (frh01, frh02, frh03, frh01_val, frh02_val, frh03_val):  # Validate validation datasets
                    validate_dataset(dataset)
            else:
                for dataset in (frh01, frh02, frh03):
                    dataset.include_area = True
                if args.OHIT:
                    synth.include_area = True

                print("Updated class IDs mapping to maintain original order:\n", new_id_mapping)

                class_modes, bottom_percent_thresholds = calculate_modes_and_thresholds([frh01_csv, frh02_csv, frh03_csv], gdf, percent=10)
                print(f"Sample class modes: {dict(list(class_modes.items())[:5])}")
                print(f"Sample bottom percent thresholds: {dict(list(bottom_percent_thresholds.items())[:5])}")

                frh01.area = area
                frh02.area = area
                frh03.area = area

                if args.max_weights_csv:
                    max_weights_per_class = load_max_weights(args.max_weights_csv, new_id_mapping)
                else:
                    max_weights_per_class = {class_id: args.max_small_parcel_weight for class_id in new_id_mapping.values() if class_id != -1}

                print("Final max_weights_per_class:", max_weights_per_class)  # Debug print
                for dataset in (frh01, frh02, frh03):
                    validate_dataset(dataset)


        if args.use_area_feature:
            
            area_class_df = gdf.set_index('id')[['area', 'classid']]

            # Apply logarithmic transformation and min-max scaling
            log_area = np.log1p(area)
            min_area = log_area.min()
            max_area = log_area.max()
            area_normalized = (log_area - min_area) / (max_area - min_area)

            if args.exclude_area_feature_for_classes:
                # Directly mask the area for excluded classes
                for class_id in args.exclude_area_feature_for_classes:
                    area_normalized[area_class_df['classid'] == class_id] = 0

            # Assign the normalized area to the dataset
            for dataset in (frh01, frh02, frh03) + ((synth,) if args.OHIT else ()):
                dataset.area_normalized = area_normalized
                dataset.use_area_feature = args.use_area_feature

            print("Sanity check: Mean normalized area per class (after exclusion logic if applied)")

            # Merge the normalized area back into the area_class_df for groupby operation
            area_class_df['normalized_area'] = area_normalized

            # Calculate mean normalized area for each class using the updated DataFrame
            class_mean_areas = area_class_df.groupby('classid')['normalized_area'].mean()

            # Print the mean normalized area for each class
            for class_id, mean_area in class_mean_areas.items():
                print(f"Class (ID: {class_id}): Mean Normalized Area = {mean_area:.4f}")

    #  modify ndims if the area feature is used
    if args.use_area_feature:
        meta['ndims'] += 1  # Adding one more dimension for the area feature
    if args.exclude_bands:
        meta['ndims'] -= len(args.exclude_bands) 


    if args.OHIT:
        datasets = (frh01,frh02,frh03,synth)
        
    else:
        datasets = (frh01,frh02,frh03)


    return traindataloader, validdataloader, traindatasets, meta, datasets, class_modes, bottom_percent_thresholds, max_weights_per_class, new_id_mapping

def get_model(modelname, ndims, num_classes, sequencelength, device, **hyperparameter):
    modelname = modelname.lower() #make case invariant

    if modelname in ["transformerencoder","transformer"]:
        model = TransformerModel(
            input_dim=ndims,
            num_classes=num_classes,
            d_model=64,
            n_head=1,
            n_layers=3,
            d_inner=128,
            activation="relu",
            dropout=0.4,
            **hyperparameter
            ).to(device)

    else:
        raise ValueError("invalid model argument. choose 'TransformerEncoder'")

    return model

def metrics(y_true, y_pred, num_classes, class_names):
    class_names = [name.replace(" ", "_") for name in class_names]
    
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    kappa = sklearn.metrics.cohen_kappa_score(y_true, y_pred)
    f1_micro = sklearn.metrics.f1_score(y_true, y_pred, average="micro")
    f1_macro = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
    f1_weighted = sklearn.metrics.f1_score(y_true, y_pred, average="weighted")
    recall_micro = sklearn.metrics.recall_score(y_true, y_pred, average="micro")
    recall_macro = sklearn.metrics.recall_score(y_true, y_pred, average="macro")
    recall_weighted = sklearn.metrics.recall_score(y_true, y_pred, average="weighted")
    precision_micro = sklearn.metrics.precision_score(y_true, y_pred, average="micro")
    precision_macro = sklearn.metrics.precision_score(y_true, y_pred, average="macro")
    precision_weighted = sklearn.metrics.precision_score(y_true, y_pred, average="weighted")
    gmean = geometric_mean_score(y_true, y_pred)
    

    result_dict = {
        "accuracy": accuracy,
        "kappa": kappa,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "recall_micro": recall_micro,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "precision_micro": precision_micro,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "gmean": gmean,
    }

    # class-wise precision and recall
    precision_per_class = precision_score(y_true, y_pred, average=None, labels=range(num_classes))
    recall_per_class = recall_score(y_true, y_pred, average=None, labels=range(num_classes))


    
    # Ensure class_names is a list or array of class names ordered by class ID
    assert len(class_names) == num_classes, "class_names length must match num_classes"

    precision_per_class_dict = {f'precision_{class_names[i]}': p for i, p in enumerate(precision_per_class)}
    recall_per_class_dict = {f'recall_{class_names[i]}': r for i, r in enumerate(recall_per_class)}

    result_dict.update(precision_per_class_dict)
    result_dict.update(recall_per_class_dict)

    return result_dict


def train_epoch(model, optimizer, criterion, dataloader, device, args, meta):
    model.train()
    losses = list()
    with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
        for idx, batch in iterator:
            optimizer.zero_grad()
            if args.loss == 'doubleObjectiveWeightedCrossEntropy':
                # Corrected for batches that contain (x, y_true, area, row_id)
                assert len(batch) == 4, "Batch must contain (x, y_true, area, row_id)"
                x, y_true, area, _ = batch  # Unpack and ignore row_id with _
                loss = criterion(model.forward(x.to(device)), y_true.to(device), area.to(device))

            else:
                # Corrected for batches when not using doubleObjectiveWeightedCrossEntropy
                # Since row_id is always included
                assert len(batch) == 3, "Batch must contain (x, y_true, row_id)"
                x, y_true, _ = batch  # Unpack and ignore row_id with _
                expected_feature_dim = meta['ndims']  # Use meta['ndims'] for expected feature dimension
                assert x.shape[-1] == expected_feature_dim, f"Expected feature dimension: {expected_feature_dim}, but got: {x.shape[-1]}"

                loss = criterion(model.forward(x.to(device)), y_true.to(device))



            loss.backward()
            optimizer.step()
            iterator.set_description(f"train loss={loss:.2f}")
            losses.append(loss)
    return torch.stack(losses)



def test_epoch(model, criterion, dataloader, device):
    model.eval()
    with torch.no_grad():
        losses = list()
        y_true_list = list()
        y_pred_list = list()
        y_score_list = list()
        field_ids_list = list()
        with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
            for idx, batch in iterator:
                if isinstance(criterion, doubleObjectiveWeightedCrossEntropy):
                    # Assuming the batch format for doubleObjectiveWeightedCrossEntropy is (x, y_true, area)
                    assert len(batch) == 4, "Batch must contain (x, y_true, area)"
                    x, y_true, area, field_id = batch
                    logprobabilities = model.forward(x.to(device))
                    loss = criterion(logprobabilities, y_true.to(device), area.to(device))
                else:
                    assert len(batch) == 3, "Batch must contain (x, y_true, field_id)"
                    # Standard batch format for other loss types
                    x, y_true, field_id = batch[:3]  # Safely unpack first two elements
                    logprobabilities = model.forward(x.to(device))
                    loss = criterion(logprobabilities, y_true.to(device))

                iterator.set_description(f"test loss={loss:.2f}")
                losses.append(loss)
                y_true_list.append(y_true)
                y_pred_list.append(logprobabilities.argmax(-1))
                y_score_list.append(logprobabilities.exp())
                field_ids_list.append(batch[-1])  # Assuming field_id is always the last element in the batch
        return torch.stack(losses), torch.cat(y_true_list), torch.cat(y_pred_list), torch.cat(y_score_list), torch.cat(field_ids_list)


def parse_args():
    parser = argparse.ArgumentParser(description='Train an evaluate time series deep learning models on the'
                                                 'BreizhCrops dataset. This script trains a model on training dataset'
                                                 'partition, evaluates performance on a validation or evaluation partition'
                                                 'and stores progress and model paths in --logdir')
    parser.add_argument(
        '--model', type=str, default="TransformerEncoder", help='select model architecture. Available models are: "TransformerEncoder"')
    parser.add_argument(
        '-b', '--batchsize', type=int, default=1024, help='batch size (number of time series processed simultaneously)')
    parser.add_argument(
        '-e', '--epochs', type=int, default=300, help='number of training epochs (training on entire dataset)')   # IN paper, 26 epochs
    parser.add_argument(
        '-m', '--mo de', type=str, default="evaluation", help='training mode. Either "validation" '
                                                             '(train on FRH01+FRH02 test on FRH03) or '
                                                             '"evaluation" (train on FRH01+FRH02+FRH03 )')
    parser.add_argument(
        '-D', '--datapath', type=str, default="../data", help='directory to download and store the dataset')
    parser.add_argument(
        '-w', '--workers', type=int, default=0, help='number of CPU workers to load the next batch')
    parser.add_argument(
        '-H', '--hyperparameter', type=str, default =None, help='model specific hyperparameter as single string, ''separated by comma of format param1=value1,param2=value2')
    parser.add_argument(
        '--level', type=str, default="L1C", help='Sentinel 2 processing level (L1C, L2A)')
    parser.add_argument(
        '--weight-decay', type=float, default=5.51e-8, help='optimizer weight_decay ')
    parser.add_argument(
        '--learning-rate', type=float, default=1.31e-3, help='optimizer learning rate ')
    parser.add_argument(
        '--preload-ram', action='store_true', help='load dataset into RAM upon initialization')
    parser.add_argument(
        '-d', '--device', type=str, default=None, help='torch.Device. either "cpu" or "cuda". '
                                                       'default will check by torch.cuda.is_available() ')
    parser.add_argument(
        '-l', '--logdir', type=str, default="/tmp", help='logdir to store progress and models (defaults to /tmp)')
    parser.add_argument(
        '-c', '--csv-folder', type=str, required=True, help='Path to the folder containing the custom CSV files for the datasets')
    parser.add_argument('--loss', type=str, default='default', choices=['default', 'weightedCrossEntropy','doubleObjectiveWeightedCrossEntropy', 'focal_loss'], help='Loss type.')
    parser.add_argument('--focal-loss-gamma', type=float, default=None, help='Gamma value for focal loss (default is 2.0).')
    parser.add_argument('--max-small-parcel-weight', type=float, default=None, help='Maximum weight for small parcels in the DOWCE')
    parser.add_argument('--max-weights-csv', type=str, help='Path to CSV file with class-specific max weights')

    parser.add_argument(
        '--OHIT', action='store_true', help='Include synthetic samples if set. Default: False.')
    
    parser.add_argument('--use-area-feature', action='store_true', help='Include area feature in the input feature space')
    parser.add_argument('--exclude-area-feature-for-classes', nargs='*', type=int, default=[], help='List of class IDs to exclude the area feature for')
    parser.add_argument('--exclude-bands', nargs='*', type=str, default=None, help='List of bands to exclude from the dataset')
    parser.add_argument('--RO', action='store_true', help='Use random oversampled dataset with is_val column for splitting')
    parser.add_argument('--checkpoint', action='store_true', help='Resume training from the last saved checkpoint if available.')

    args = parser.parse_args()

    hyperparameter_dict = dict()
    if args.hyperparameter is not None:
        for hyperparameter_string in args.hyperparameter.split(","):
            param, value = hyperparameter_string.split("=")
            hyperparameter_dict[param] = float(value) if '.' in value else int(value)
    args.hyperparameter = hyperparameter_dict

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


if __name__ == "__main__":
    args = parse_args()
    set_seed(seed_value=42)
    train(args)
