import geopandas as gpd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
import numpy as np
import os
from sklearn.metrics import confusion_matrix


def analyze_attribute_influence(gdf, attribute, num_bins, folder_name = None):

    gdf['True_Label'] = gdf['True_Label'].astype(int)
    gdf['Predicted_Label'] = gdf['Predicted_Label'].astype(int)
    
    # Ensure the attribute column is numeric, coercing errors to NaN
    gdf[attribute] = pd.to_numeric(gdf[attribute], errors='coerce')

    # Mapping from class IDs to class names
    class_id_to_name = dict(zip(gdf['True_Label'], gdf['classname']))
    
    training_set = gdf[gdf['Dataset'].isin(['frh01', 'frh02', 'frh03'])]
    testing_set = gdf[gdf['Dataset'] == 'frh04']
    
    unique_classes = np.unique(gdf['True_Label'])
    
    last_folder_name = os.path.basename(os.path.normpath(folder_name))
    base_dir = os.path.join(folder_name, 'sensitive_attribute_heatmaps', attribute)
    os.makedirs(base_dir, exist_ok=True)

    def calculate_metrics(data, attribute, bins, unique_classes):
        # Ensure numeric conversion for 'Predicted_Label' and 'True_Label'
        data['True_Label'] = pd.to_numeric(data['True_Label'], errors='coerce').astype(int)
        data['Predicted_Label'] = pd.to_numeric(data['Predicted_Label'], errors='coerce').astype(int)

        results = []
        for i in range(len(bins) - 1):
            bin_data = data[(data[attribute] >= bins[i]) & (data[attribute] < bins[i + 1])]

            for class_id in unique_classes:
                TP = ((bin_data['Predicted_Label'] == class_id) & (bin_data['True_Label'] == class_id)).sum()
                FP = ((bin_data['Predicted_Label'] == class_id) & (bin_data['True_Label'] != class_id)).sum()
                FN = ((bin_data['Predicted_Label'] != class_id) & (bin_data['True_Label'] == class_id)).sum()

                precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
                recall = TP / (TP + FN) if (TP + FN) > 0 else np.nan

                results.append({
                    'bin_start': int(bins[i]),
                    'bin_range': f'{bins[i]}-{bins[i + 1]}',
                    'class_id': class_id,  # Use class ID here for mapping later
                    'precision': precision,
                    'recall': recall
                })

            if not bin_data.empty:
                global_precision = precision_score(bin_data['True_Label'], bin_data['Predicted_Label'], average='macro', zero_division=0)
                global_recall = recall_score(bin_data['True_Label'], bin_data['Predicted_Label'], average='macro', zero_division=0)

                results.append({
                    'bin_start': int(bins[i]),
                    'bin_range': f'{bins[i]}-{bins[i + 1]}',
                    'class_id': 'Macro Average per Bin',  # Use a string to differentiate from actual class IDs
                    'precision': global_precision,
                    'recall': global_recall
                })

        metrics_df = pd.DataFrame(results)

        # Map class IDs to class names
        metrics_df['class'] = metrics_df['class_id'].map(class_id_to_name).fillna(metrics_df['class_id'])
        
        # Ensure sorting by 'bin_start' as a numeric value
        metrics_df.sort_values(by='bin_start', inplace=True)

        # Drop 'bin_start' and 'class_id' columns, and calculate macro averages across bins
        macro_avg_across_bins = metrics_df[metrics_df['class'] != 'Macro Average per Bin'].groupby('class').mean(numeric_only=True).reset_index()
        macro_avg_across_bins['bin_range'] = 'Macro Average Across Bins'

        # Append macro averages to the DataFrame, ensuring final structure
        final_df = pd.concat([metrics_df, macro_avg_across_bins], ignore_index=True)

        return final_df

    def plot_and_save_heatmaps_and_csv(metrics_df, title_prefix):
        plot_base_dir = os.path.join(base_dir, title_prefix.replace(' ', '_'))
        os.makedirs(plot_base_dir, exist_ok=True)

        metrics_df['class'] = metrics_df['class'].astype(str)  # Ensure class is treated as a string for plotting

        # Prepare a DataFrame for plotting that includes both 'bin_range' and 'order' for correct sorting
        unique_bins = metrics_df[['bin_start', 'bin_range']].drop_duplicates().sort_values('bin_start')
        bin_order = {row['bin_range']: i for i, row in unique_bins.iterrows()}

        # Use 'bin_range' for indexing in the pivot table to maintain the correct order
        metrics_df['bin_order'] = metrics_df['bin_range'].map(bin_order)

        for metric in ['precision', 'recall']:
            # Pivot the DataFrame using 'bin_order' to sort, then replace the index with 'bin_range'
            metric_df = metrics_df.pivot(index='bin_order', columns='class', values=metric)
            metric_df.index = [unique_bins.loc[idx, 'bin_range'] for idx in metric_df.index]

            # Ensure 'Macro Average per Bin' is the last column
            # Check if 'Macro Average per Bin' exists in the columns, then move it to the end
            if 'Macro Average per Bin' in metric_df.columns:
                # Reorder columns to move 'Macro Average per Bin' to the end
                cols = [col for col in metric_df.columns if col != 'Macro Average per Bin'] + ['Macro Average per Bin']
                metric_df = metric_df[cols]

            plt.figure(figsize=(14, 10))  # Adjusted figure size for potential additional rows
            sns.heatmap(metric_df, annot=True, cmap='viridis', fmt='.2f')
            metric_title = f'{title_prefix}_{metric}_by_{attribute}_Bin_and_Class_for_{last_folder_name}'.replace(' ', '_')
            plt.title(metric_title.replace('_', ' '))
            plt.ylabel(f'{attribute} Bin')
            plt.xlabel('Class')
            plt.tight_layout()  # Adjust layout to fit everything
            heatmap_path = os.path.join(plot_base_dir, f'{metric_title}.png')
            plt.savefig(heatmap_path)
            plt.close()

            # Save corresponding CSV
            csv_path = heatmap_path.replace('.png', '.csv')
            metric_df.reset_index().to_csv(csv_path, index=False)


    # Calculate bin edges for training and testing sets separately
    _, training_bin_edges = pd.qcut(training_set[attribute], q=num_bins, retbins=True, duplicates='drop')
    _, testing_bin_edges = pd.qcut(testing_set[attribute], q=num_bins, retbins=True, duplicates='drop')

    # Adjust the rightmost bin edge to include the maximum value
    training_bin_edges[-1] = training_set[attribute].max() + 1
    testing_bin_edges[-1] = testing_set[attribute].max() + 1

    # Generate labels for each bin based on the bin edges
    training_bin_labels = [f"{int(edge_left)}-{int(edge_right)}" for edge_left, edge_right in zip(training_bin_edges[:-1], training_bin_edges[1:])]
    testing_bin_labels = [f"{int(edge_left)}-{int(edge_right)}" for edge_left, edge_right in zip(testing_bin_edges[:-1], testing_bin_edges[1:])]

    # Now pass the bin edges to the calculate_metrics function for training and testing
    training_metrics = calculate_metrics(training_set, attribute, training_bin_edges, unique_classes)
    testing_metrics = calculate_metrics(testing_set, attribute, testing_bin_edges, unique_classes)
    
    plot_and_save_heatmaps_and_csv(training_metrics, 'Training_Set')
    plot_and_save_heatmaps_and_csv(testing_metrics, 'Testing_Set')


def generate_confusion_matrix(gdf, folder_name):
    # Extract the parent folder's name
    parent_folder_name = os.path.basename(folder_name)
    # Create the output directory for the confusion matrix
    output_dir = os.path.join(folder_name, 'confusion_matrix')
    os.makedirs(output_dir, exist_ok=True)
    
    # Split the data into training and testing datasets
    training_data = gdf[gdf['region'] != 'frh04']
    testing_data = gdf[gdf['region'] == 'frh04']
    
    def save_confusion_matrix(data, dataset_type):
        # Calculate the confusion matrix
        true_labels = data['True_Label'].astype(int)
        predicted_labels = data['Predicted_Label'].astype(int)
        
        # Print unique labels for diagnostics
        print(f"Unique true labels for {dataset_type}: {np.unique(true_labels)}")
        print(f"Unique predicted labels for {dataset_type}: {np.unique(predicted_labels)}")
        
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Ensure unique true and predicted labels are of the same length
        unique_true_labels = np.unique(true_labels)
        unique_predicted_labels = np.unique(predicted_labels)
        unique_labels = np.union1d(unique_true_labels, unique_predicted_labels)
        
        # Print combined unique labels for diagnostics
        print(f"Combined unique labels for {dataset_type}: {unique_labels}")
        
        # Convert the confusion matrix to a DataFrame for easy saving
        cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
        
        # Define file names with the parent folder's name and dataset type (training/testing)
        base_filename = f'{dataset_type}_confusion_matrix_{parent_folder_name}'
        csv_path = os.path.join(output_dir, f'{base_filename}.csv')
        png_path = os.path.join(output_dir, f'{base_filename}.png')
        svg_path = os.path.join(output_dir, f'{base_filename}.svg')
        
        # Save the confusion matrix as a CSV file
        cm_df.to_csv(csv_path)
        
        # Plot the confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {parent_folder_name} ({dataset_type})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Save the confusion matrix plot as PNG
        plt.savefig(png_path, dpi=150)
        
        # Save the confusion matrix plot as SVG
        plt.savefig(svg_path, format='svg')
        
        # Close the plot to free up memory
        plt.close()
        
        return csv_path, png_path, svg_path
    
    # Save confusion matrices for training and testing datasets
    training_paths = save_confusion_matrix(training_data, 'training')
    testing_paths = save_confusion_matrix(testing_data, 'testing')
    
    return training_paths, testing_paths

def save_statistics_summary(folder_path):
    def summarize_training_log(csv_file_path, early_stopping_metric='testloss'):
        folder_title = os.path.basename(os.path.dirname(csv_file_path))
        
        # Create summary_statistics folder inside folder_path
        summary_folder_path = os.path.join(folder_path, 'summary_statistics')
        os.makedirs(summary_folder_path, exist_ok=True)

        # Read the CSV file
        data = pd.read_csv(csv_file_path)

        # Replace inf values with NaN and drop rows with missing values
        data = data.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any', subset=['trainloss', 'valloss'])

        # Check if 'epoch' column exists, if not, create it
        if 'epoch' not in data.columns:
            data['epoch'] = range(1, len(data) + 1)

        # Find the epoch with the highest gmean and its details
        if 'gmean' in data.columns and not data['gmean'].isna().all():
            max_gmean_row = data.loc[data['gmean'].idxmax()]
        else:
            max_gmean_row = pd.Series()

        # Generate statistical summary table
        stats_df = data.describe().transpose()
        stats_df.drop(columns='count', inplace=True, errors='ignore')
        for col in data.columns:
            if col in stats_df.index:
                stats_df.at[col, 'Highest GMean Epoch'] = max_gmean_row.get(col, np.nan)
        
        # Reorder columns to have 'Highest GMean Epoch' as the first column
        cols = list(stats_df.columns)
        cols.insert(0, cols.pop(cols.index('Highest GMean Epoch')))
        stats_df = stats_df[cols]

        # Save statistical summary to CSV in summary_statistics folder
        summary_csv_path = os.path.join(summary_folder_path, f'{folder_title}_val_statistics.csv')
        stats_df.to_csv(summary_csv_path)

        return stats_df

    # Process 'trainlog.csv' in the specified folder
    csv_file_path = os.path.join(folder_path, 'trainlog.csv')
    if os.path.isfile(csv_file_path):
        statistics_summary_df = summarize_training_log(csv_file_path)
    else:
        print(f"No 'trainlog.csv' found in {folder_path}")
        return
    
    summary_folder_path = os.path.join(folder_path, 'summary_statistics')
    folder_title = os.path.basename(os.path.dirname(csv_file_path))

    # Process the test metrics CSV
    test_metrics_file_path = os.path.join(folder_path, os.path.basename(os.path.normpath(folder_path)) + '_results_test_metrics.csv')
    if os.path.isfile(test_metrics_file_path):
        test_metrics_df = pd.read_csv(test_metrics_file_path)

        # Transpose the test metrics dataframe
        test_metrics_transposed = test_metrics_df.transpose()
        test_metrics_transposed.columns = ['test_Highest GMean Epoch']
        test_metrics_transposed.index.name = 'Metric'

        # Prepare the statistics summary DataFrame
        statistics_summary_df = statistics_summary_df.reset_index().rename(columns={'index': 'Metric'})
        
        # Merge the transposed test metrics with the statistics summary
        merged_df = pd.merge(test_metrics_transposed, statistics_summary_df, left_index=True, right_on='Metric', how='outer')

        # Rename the column for validation highest gmean epoch
        merged_df = merged_df.rename(columns={'Highest GMean Epoch': 'val_Highest GMean Epoch'})

        # Prefix columns related to validation with 'val_'
        val_columns = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
        merged_df = merged_df.rename(columns={col: f'val_{col}' for col in val_columns})

        # Add fairness metrics to the test_Highest GMean Epoch
        fairness_metrics_path = os.path.join(folder_path, 'fairness_metrics')
        if os.path.isdir(fairness_metrics_path):
            for file_name in os.listdir(fairness_metrics_path):
                if file_name.startswith('testing_fairness') and file_name.endswith('.csv'):
                    fairness_file_path = os.path.join(fairness_metrics_path, file_name)
                    fairness_data = pd.read_csv(fairness_file_path)
                    if 'Metric' in fairness_data.columns and 'Value' in fairness_data.columns:
                        for _, row in fairness_data.iterrows():
                            metric = row['Metric']
                            value = row['Value']
                            if metric in merged_df['Metric'].values:
                                merged_df.loc[merged_df['Metric'] == metric, 'test_Highest GMean Epoch'] = value
                            else:
                                new_row = pd.DataFrame({'Metric': [metric], 'test_Highest GMean Epoch': [value]})
                                merged_df = pd.concat([merged_df, new_row], ignore_index=True)

        # Reset index to make 'Metric' a column again
        merged_df = merged_df.set_index('Metric').reset_index()

        # Save the merged dataframe to CSV
        output_path = os.path.join(summary_folder_path, f'{folder_title}_summary_statistics.csv')
        merged_df.to_csv(output_path, index=False)

        print(f"Combined summary statistics saved to {output_path}")
        return output_path
    else:
        print(f"No '{os.path.basename(folder_path)}_results_test_metrics.csv' found in {folder_path}")
        return


def calculate_fairness_metrics(gdf, folder_name):
    # Create folder if it doesn't exist
    output_folder = os.path.join(folder_name, "fairness_metrics")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Extract the suffix from the folder name
    folder_suffix = os.path.basename(folder_name)
    
    # Split the data into training and testing datasets
    training_data = gdf[gdf['region'] != 'frh04']
    testing_data = gdf[gdf['region'] == 'frh04']
    
    # Define privileged and unprivileged groups based on parcel area
    privileged_group = gdf['area'] > 6170
    unprivileged_group = gdf['area'] <= 6170
    
    def compute_metrics(data, group, label, predicted_label):
        # Confusion matrix components for privileged and unprivileged groups
        correct_priv = np.sum((data[label] == data[predicted_label]) & group)
        incorrect_priv = np.sum((data[label] != data[predicted_label]) & group)
        correct_unpriv = np.sum((data[label] == data[predicted_label]) & ~group)
        incorrect_unpriv = np.sum((data[label] != data[predicted_label]) & ~group)
        
        # Rates
        TPR_priv = correct_priv / (correct_priv + incorrect_priv)
        TPR_unpriv = correct_unpriv / (correct_unpriv + incorrect_unpriv)
        
        FPR_priv = incorrect_priv / (incorrect_priv + correct_priv)
        FPR_unpriv = incorrect_unpriv / (incorrect_unpriv + correct_unpriv)
        
        # Accuracy
        accuracy_priv = correct_priv / (correct_priv + incorrect_priv)
        accuracy_unpriv = correct_unpriv / (correct_unpriv + incorrect_unpriv)
        
        # Metrics calculations
        metrics = {
            'Statistical Parity Difference': (correct_unpriv / len(data[~group])) - (correct_priv / len(data[group])),
            'Error Rate Ratio': (incorrect_unpriv / len(data[~group])) / (incorrect_priv / len(data[group])),
            'Disparate Impact': (correct_unpriv / len(data[~group])) / (correct_priv / len(data[group])),
            'True Positive Rate (Privileged)': TPR_priv,
            'True Positive Rate (Unprivileged)': TPR_unpriv,
            'False Positive Rate (Privileged)': FPR_priv,
            'False Positive Rate (Unprivileged)': FPR_unpriv,
            'Accuracy (Privileged)': accuracy_priv,
            'Accuracy (Unprivileged)': accuracy_unpriv
        }
        
        return metrics

    def compute_class_metrics(data, group, label, predicted_label, class_column):
        class_metrics = {}
        for classname in data[class_column].unique():
            class_data = data[data[class_column] == classname]
            class_metrics[classname] = compute_metrics(class_data, group[data[class_column] == classname], label, predicted_label)
        return class_metrics
    
    # Calculate overall metrics for training data
    training_metrics = compute_metrics(training_data, privileged_group[training_data.index], 'True_Label', 'Predicted_Label')
    
    # Calculate overall metrics for testing data
    testing_metrics = compute_metrics(testing_data, privileged_group[testing_data.index], 'True_Label', 'Predicted_Label')
    
    # Calculate class metrics for training data
    training_class_metrics = compute_class_metrics(training_data, privileged_group[training_data.index], 'True_Label', 'Predicted_Label', 'classname')
    
    # Calculate class metrics for testing data
    testing_class_metrics = compute_class_metrics(testing_data, privileged_group[testing_data.index], 'True_Label', 'Predicted_Label', 'classname')
    
    # Save overall results to CSV files
    training_metrics_df = pd.DataFrame(list(training_metrics.items()), columns=['Metric', 'Value'])
    testing_metrics_df = pd.DataFrame(list(testing_metrics.items()), columns=['Metric', 'Value'])
    
    training_metrics_df.to_csv(os.path.join(output_folder, f'training_fairness_metrics_{folder_suffix}.csv'), index=False)
    testing_metrics_df.to_csv(os.path.join(output_folder, f'testing_fairness_metrics_{folder_suffix}.csv'), index=False)
    
    # Save class metrics results to CSV files
    training_class_metrics_df = pd.DataFrame(training_class_metrics).T.reset_index().rename(columns={'index': 'Classname'})
    testing_class_metrics_df = pd.DataFrame(testing_class_metrics).T.reset_index().rename(columns={'index': 'Classname'})
    
    training_class_metrics_df.to_csv(os.path.join(output_folder, f'training_fairness_metrics_per_class_{folder_suffix}.csv'), index=False)
    testing_class_metrics_df.to_csv(os.path.join(output_folder, f'testing_fairness_metrics_per_class_{folder_suffix}.csv'), index=False)
    
    return training_metrics_df, testing_metrics_df, training_class_metrics_df, testing_class_metrics_df
