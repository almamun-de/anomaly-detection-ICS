


def main(csv_file_paths, scenario, classifiers, feature_name):
    # Load the CSV file containing predicted and actual labels
    data_lst = []
    for csv_file_path, classifier in zip(csv_file_paths, classifiers):
        predicted_data = pd.read_csv(csv_file_path)

        if scenario == 1:
            # Convert all non-zero labels to 1
            predicted_data.loc[predicted_data['label'] != 0, 'label'] = 1
            if -1 in predicted_data['predicted'].to_numpy():
                # Convert predicted labels of 1 to 0. Because for Unsupervised model, predicted == 1 means no attack predicted.
                predicted_data.loc[predicted_data['predicted'] == 1, 'predicted'] = 0
                # Convert predicted labels of -1 to 1. Because for Unsupervised model, predicted == -1 means attack predicted.
                predicted_data.loc[predicted_data['predicted'] == -1, 'predicted'] = 1

            # Calculate precision and recall
            precision = precision_score(predicted_data['label'], predicted_data['predicted'], average='binary', zero_division=0)
            recall = recall_score(predicted_data['label'], predicted_data['predicted'], average='binary', zero_division=0)

        elif scenario == 2:
            # Calculate precision and recall
            precision = sum(precision_score(predicted_data['label'], predicted_data['predicted'], average=None, zero_division=0)[2:])/51
            recall = sum(recall_score(predicted_data['label'], predicted_data['predicted'], average=None, zero_division=0)[2:])/51

        elif scenario == 3:
            predicted_data.loc[predicted_data['label'] != 0, 'label'] = 1
            if -1 in predicted_data['predicted'].to_numpy():
                # Convert predicted labels of 1 to 0. Because for Unsupervised model, predicted == 1 means no attack predicted.
                predicted_data.loc[predicted_data['predicted'] == 1, 'predicted'] = 0
                # Convert predicted labels of -1 to 1. Because for Unsupervised model, predicted == -1 means attack predicted.
                predicted_data.loc[predicted_data['predicted'] == -1, 'predicted'] = 1

                # Convert all non-zero labels to 1

                # Calculate precision and recall
            precision = precision_score(predicted_data['label'], predicted_data['predicted'], average='binary', zero_division=0)
            recall = recall_score(predicted_data['label'], predicted_data['predicted'], average='binary', zero_division=0)

        data_lst.append((classifier, 'Precision', precision))
        data_lst.append((classifier, 'Recall', recall))
        #else:
            #precision = precision_score(predicted_data['label'], predicted_data['predicted'], average=None, zero_division=0)[1]
            #recall = recall_score(predicted_data['label'], predicted_data['predicted'], average=None, zero_division=0)[1]

    data = {}
    for category, subcategory, value in data_lst:
        if category not in data:
            data[category] = {}  # Initialize a new dictionary for this category
        # Assign the value to the subcategory within this category
        data[category][subcategory] = value


    # Number of models and metrics (precision and recall)
    n_models = len(data)
    # The width of the bars
    width = 0.35  # Adjusting width for visual clarity

    # Define colors for precision and recall
    colors = {'Precision': 'skyblue', 'Recall': 'orange'}

    # Positions for each of the models
    pos = np.arange(n_models)

    fig, ax = plt.subplots()

    # Iterate over each model to plot precision and recall
    for i, (model_name, scores) in enumerate(data.items()):
        for j, (score_name, values) in enumerate(scores.items()):
            # Calculate the position for precision and recall of each model
            score_pos = pos[i] + (j * width) - width / 2
            bars = plt.bar(score_pos, values, width, label=f'{score_name}' if i == 0 else "", alpha=0.75,
                    color=colors[score_name])
            # Annotation
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

    # Adding some additional features
    ax.set_ylabel('Scores')
    ax.set_title(f"Scenario:{scenario} {feature_name} Classification Metrics")
    ax.set_xticks(pos)
    ax.set_xticklabels(data.keys())
    # Update legend to reflect the color coding
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    #ax.legend(by_label.values(), by_label.keys(), title="Metric", loc='center left', bbox_to_anchor=(-0.15, 1.07))
    ax.legend(by_label.values(), by_label.keys(), title="Metric")

    output_file = f'S{scenario}-{feature_name}-classification-metrics.png'
    plt.savefig(output_file)

    # Output the file saving information
    print(f"Plot saved as '{output_file}'.")


# Command line argument parsing setup
if __name__ == "__main__":
    # Command-line argument parsing setup
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-f", "--file", nargs='+', required=True, type=str, help='Path to the CSV file')
    argParser.add_argument("-s", "--scenario", help="The scenario to be applied.")
    argParser.add_argument("-c", "--classifier", nargs='+', required=True, type=str, help='classifiers')
    argParser.add_argument("-t", "--feature", help="feature")
    args = argParser.parse_args()

    # Execute the main function with provided arguments
    main(args.file, int(args.scenario), args.classifier, args.feature)

# Usage example:
# python plot-classifier-result.py -f path_to_csv_file.csv -s 1
