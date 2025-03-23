import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main(csv_file_path):
    # Load the CSV file containing predicted and actual labels
    predicted_data = pd.read_csv(csv_file_path)

    # Unsupervised classifiers 'predicted' == -1 means attack predicted.
    if -1 in predicted_data['predicted'].to_numpy():
        # Convert all non-zero labels to 1
        predicted_data.loc[predicted_data['label'] != 0, 'label'] = 1
        # Convert predicted labels of -1 to 1. Because for Unsupervised model, predicted == -1 means attack predicted.
        predicted_data.loc[predicted_data['predicted'] == -1, 'predicted'] = 1
        # Convert predicted labels of 1 to 0. Because for Unsupervised model, predicted == 1 means no attack predicted.
        predicted_data.loc[predicted_data['predicted'] == 1, 'predicted'] = 0

    # Calculate classification metrics
    accuracy = accuracy_score(predicted_data['label'], predicted_data['predicted'])
    precision = precision_score(predicted_data['label'], predicted_data['predicted'], average='weighted')
    recall = recall_score(predicted_data['label'], predicted_data['predicted'], average='weighted')
    f1 = f1_score(predicted_data['label'], predicted_data['predicted'], average='weighted')

    # Plotting the metrics for visual representation
    plt.figure(figsize=(10, 6))
    metrics = [accuracy, precision, recall, f1]
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    sns.barplot(x=labels, y=metrics)

    # Set plot title based on the CSV file name
    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    plt.title(f'Classification Metrics for {base_name}')
    plt.ylabel('Score')

    # Save the plot as an image file
    output_file = f'{base_name}_classification_metrics.png'
    plt.savefig(output_file)

    # Output the file saving information
    print(f"Plot saved as '{output_file}'.")



# Usage example:
# python plot-classifier-result.py -f path_to_csv_file.csv
