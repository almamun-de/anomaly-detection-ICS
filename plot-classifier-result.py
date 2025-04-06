

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


# Command line argument parsing setup
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Classifier Results')
    parser.add_argument('-f', '--file', required=True, type=str, help='Path to the CSV file')
    args = parser.parse_args()

    # Execute the main function with provided arguments
    main(args.file)

# Usage example:
# python plot-classifier-result.py -f path_to_csv_file.csv
