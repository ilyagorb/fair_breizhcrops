import sys
import os
import torch
import breizhcrops
import pandas as pd
import geopandas as gpd
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.metrics import geometric_mean_score
from breizhcrops.models import TransformerModel
import numpy as np

from analysis_functions import analyze_attribute_influence
from analysis_functions import save_statistics_summary
from analysis_functions import calculate_fairness_metrics
from analysis_functions import generate_confusion_matrix

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

def get_results(folder_name, csv_folder_path, gdf, datapath, num_classes, ndims, excluded_classes_from_area_feature, preload_ram=False, level="L1C"):

    # Load the training log
    log_df = pd.read_csv(f'{folder_name}/trainlog.csv')

    # Find the epoch with the highest G-mean value
    best_epoch = int(log_df.loc[log_df['gmean'].idxmax()]['epoch'])
    
    print(f"Best epoch: {best_epoch}")
    
    # Initialize the model with the same configuration as used during training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(
        input_dim=ndims,
        num_classes=num_classes,  
        d_model=64,
        n_head=1,
        n_layers=3,
        d_inner=128,
        activation="relu",
        dropout=0.4
    ).to(device)

    # Load the model with the best epoch's state
    model_path = f'{folder_name}/epoch_saves/model_state_epoch_{best_epoch}.pth' 
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Initialize an empty list to hold the results
    results = []
    
    dataset_names = ["frh01", "frh02", "frh03", "frh04"]

    if ndims == 14:
        area = gdf.set_index('id')['area']
        log_area = np.log1p(area)
        min_area = log_area.min()
        max_area = log_area.max()
        area_normalized = (log_area - min_area) / (max_area - min_area)
        
        if excluded_classes_from_area_feature:
            area_class_df = gdf.set_index('id')[['area', 'classid']]
            for class_id in excluded_classes_from_area_feature:
                area_normalized[area_class_df['classid'] == class_id] = 0

    for dataset_name in dataset_names:
        csv_file_path = os.path.join(csv_folder_path, f'{dataset_name}.csv')
        dataset = breizhcrops.BreizhCrops(region=dataset_name, root=datapath, preload_ram=preload_ram, level=level, csv_file_name=csv_file_path)

        if ndims == 14:
            dataset.area_normalized = area_normalized
            dataset.use_area_feature = True

        # Loop over all examples in the dataset
        for i in range(len(dataset)):
            x, y, field_id = dataset[i]  # Get features, label, and field ID
            x_batch = x.unsqueeze(0).to(device)  # Add a batch dimension and move to the same device as model
            with torch.no_grad():  # Disable gradient computation
                y_pred = model(x_batch)  # Predict
                y_pred_label = torch.argmax(y_pred, dim=1)  # Get predicted label

            # Ensure labels are scalars
            y = y.item() if isinstance(y, torch.Tensor) else y
            y_pred_label = y_pred_label.item() if isinstance(y_pred_label, torch.Tensor) else y_pred_label

            # Append results
            results.append({
                'Field_ID': field_id,
                'True_Label': y,
                'Predicted_Label': y_pred_label,
                'Correct': y == y_pred_label,
                'Dataset': dataset_name
            })

    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)

    # Extract the last part of the folder path
    directory_name = os.path.basename(folder_name)

    # Save model predictions to a CSV file
    output_csv = os.path.join(folder_name, f'{directory_name}_model_accuracies.csv')
    results_df.to_csv(output_csv, index=False)
    print(f'Model predictions saved to {output_csv}')

    # Merge CSV with GeoJSON
    try:
        csv_df = pd.read_csv(output_csv)
        merged_gdf = gdf.merge(csv_df, left_on='id', right_on='Field_ID')
        output_geojson = os.path.join(folder_name, f'{directory_name}_model_accuracies.geojson')
        merged_gdf.to_file(output_geojson, driver='GeoJSON')
        print(f"Merged GeoJSON file saved at {output_geojson}")
    except Exception as e:
        print(f"Error during merging: {e}")

    # Re-load the GeoJSON file
    gdf = gpd.read_file(output_geojson)

    # Calculate and save metrics
    results_df = results_df[results_df['Dataset'] == 'frh04']
    y_true = results_df['True_Label']
    y_pred = results_df['Predicted_Label']
    class_names = dataset.classname

    metrics_dict = metrics(y_true, y_pred, num_classes, class_names)

    # Save metrics to a CSV file
    metrics_df = pd.DataFrame([metrics_dict])
    metrics_output_csv = os.path.join(folder_name, f'{directory_name}_results_test_metrics.csv')
    metrics_df.to_csv(metrics_output_csv, index=False)
    print(f'Model metrics saved to {metrics_output_csv}')

    analyze_attribute_influence(gdf, 'area', 10, folder_name)
    analyze_attribute_influence(gdf, 'distance_to_closest_city', 10, folder_name)
    calculate_fairness_metrics(gdf, folder_name)
    generate_confusion_matrix(gdf, folder_name)

    save_statistics_summary(folder_name)
