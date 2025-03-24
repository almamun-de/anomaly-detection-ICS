

# Function to calculate permutation feature importance
def calculate_feature_importance(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    return perm_importance.importances_mean


# Function to plot feature importance and save the plot
def plot_feature_importance(importances, feature_names, title, classifier_name, scenario, csv_file_name):
    indices = np.argsort(importances)[::-1]  # sorts the importances array in descending order
    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.ylabel('Permutation Importance')
    plt.tight_layout()

    # Create a directory for plots if it does not exist
    os.makedirs('feature_importance_plots', exist_ok=True)
    # Construct the plot filename
    plot_file_name = f"{classifier_name}-Scenario-{scenario}-{os.path.basename(csv_file_name).replace('.csv', '')}.png"
    plt.savefig(os.path.join('feature_importance_plots', plot_file_name))
    plt.close()


# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Feature Importance Analysis with Scenario-Based Training')
    parser.add_argument('-f', '--csv_file', type=str, required=True, help='Path to the CSV file containing the dataset')
    parser.add_argument('-s', '--scenario', type=int, choices=[1, 2, 3], required=True,
                        help='Scenario to be used for training and testing')
    parser.add_argument('-k', '--kfold', type=int, required=True, help='Number of folds for cross-validation')
    return parser.parse_args()


# Main function to run the analysis
def run_feature_importance_analysis(classifiers, dataset_path, scenario, k, csv_file_base_name):
    dataset = pd.read_csv(dataset_path)

    # Extract feature columns and the target variable
    feature_cols = [col for col in dataset.columns if col.startswith('feature')]
    X = dataset[feature_cols].to_numpy()
    y = dataset['label'].to_numpy()
    folds = dataset['fold'].to_numpy()

    for classifier_name, classifier in classifiers.items():
        print(f"Processing {classifier_name}...")
        results = []

        for i in range(k):
            # Scenario-based splitting of the dataset
            if scenario == 1:
                train_index = (folds != i + 1) & (y == 0)
            elif scenario == 2:
                train_index = (folds != i + 1) & (y != 1)
            elif scenario == 3:
                train_index = (folds != i + 1) & (y <= 1)

            test_index = (folds == i + 1)

            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

            if len(np.unique(y_train)) > 1:  # Check if y_train has more than one class
                # Calculate feature importance
                importances = calculate_feature_importance(classifier, X_train, y_train, X_test, y_test)
                results.append(importances)
            else:
                print(f"Skipping fold {i+1} for {classifier_name} due to insufficient class representation.")

        if results:  # Check if there were any valid results
            # Average the feature importance over all valid folds
            avg_importances = np.mean(results, axis=0)

            # Plotting feature importance and saving the plot
            plot_title = f"Feature Importance - {classifier_name} - Scenario {scenario}"
            plot_feature_importance(avg_importances, feature_cols, plot_title, classifier_name, scenario, csv_file_base_name)
        else:
            print(f"No valid results for {classifier_name} in Scenario {scenario}. Plotting skipped.")


if __name__ == '__main__':
    args = parse_args()
    '''
    One-Class Classifiers: One-Class SVM, EE, and LOF.
    These classifiers are not inherently provide feature importance.
    The permutation importance may not provide meaningful insights for these models.
    '''
    classifiers = {
        # Multi-class Classifiers
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'SVM': SVC(kernel='rbf', gamma='auto', random_state=42)  # Multi-class SVM

        # One-class Classifiers
        #'One-Class SVM': OneClassSVM(kernel='rbf', gamma='auto'),
        #'Elliptic Envelope': EllipticEnvelope(random_state=42),
        #'Local Outlier Factor': LocalOutlierFactor(n_neighbors=20)
    }

    # Extract the base name of the input CSV file for naming the plots
    csv_file_base_name = os.path.basename(args.csv_file)

    # Run the analysis
    run_feature_importance_analysis(classifiers, args.csv_file, args.scenario, args.kfold, csv_file_base_name)


"""
python feature-importance.py -f /path_to_csv_file.csv -s 2 -k 10
"""
