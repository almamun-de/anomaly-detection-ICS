

def model_selection(model):
    if model == 'OneClassSVM':
        classifier = OneClassSVM(gamma='auto')
        param_grid = {'nu':[0.1,0.5,0.9]}

    elif model == 'MultiClassSVM':
        classifier = SVC()
        param_grid = {'kernel':['linear', 'rbf'], 'C':[1, 10]}
        
    elif model == 'LOF':
        classifier = LocalOutlierFactor(novelty=True)
        param_grid = {'n_neighbors':[15,17,19,21,23]}

    elif model == 'EE':
        classifier = EllipticEnvelope()
        param_grid = {'contamination':[0.1,0.15,0.2,0.25]}

    elif model == 'RF':
        classifier = RandomForestClassifier()
        param_grid = {'n_estimators':[90,100,110],'criterion':['gini', 'entropy', 'log_loss']}
        
    elif model == 'KNN':
        classifier = KNeighborsClassifier()
        param_grid = {'n_neighbors':[5,7,9,11,13]}

    return classifier, param_grid

def run_experiment(classifier, param_grid, dataset_df, scenario, k):
    start_time = time.time()  # Start time
    initial_memory = memory_usage()[0]  # Initial memory usage
    clf = GridSearchCV(classifier, param_grid, scoring='accuracy')
    results = []

    ex_columns_arr = dataset_df[dataset_df.columns[dataset_df.columns.isin(['timestamp','label','fold'])]].sort_values(['fold','timestamp']).to_numpy()
    ex_columns = pd.DataFrame(ex_columns_arr,columns=['timestamp','label','fold'])


    if scenario == 1:
        for i in range(k):
            # Selecting train and test data according to the scenario.
            X_train = dataset_df[dataset_df.columns[~dataset_df.columns.isin(['timestamp', 'label', 'fold'])]][
                dataset_df['fold'] != i + 1][dataset_df['label'] == 0].to_numpy()
            y_train = dataset_df['label'][dataset_df['fold'] != i + 1][dataset_df['label'] == 0].to_numpy()
            X_test = dataset_df[dataset_df.columns[~dataset_df.columns.isin(['timestamp', 'label', 'fold'])]][
                dataset_df['fold'] == i + 1].to_numpy()
            y_test = dataset_df['label'][dataset_df['fold'] == i + 1].to_numpy()

            # Training and Testing of the classifier
            clf.fit(X_train, y_train)
            pred_test = clf.predict(X_test)
            # Store the results
            results.append(pred_test)

    elif scenario == 2:
        for i in range(k):
            X_train = dataset_df[dataset_df.columns[~dataset_df.columns.isin(['timestamp', 'label', 'fold'])]][
                dataset_df['fold'] != i + 1][dataset_df['label'] != 1].to_numpy()
            y_train = dataset_df['label'][dataset_df['fold'] != i + 1][dataset_df['label'] != 1].to_numpy()
            X_test = dataset_df[dataset_df.columns[~dataset_df.columns.isin(['timestamp', 'label', 'fold'])]][
                dataset_df['fold'] == i + 1].to_numpy()
            y_test = dataset_df['label'][dataset_df['fold'] == i + 1].to_numpy()

            clf.fit(X_train, y_train)
            pred_test = clf.predict(X_test)
            results.append(pred_test)

    elif scenario == 3:
        for i in range(k):
            X_train = dataset_df[dataset_df.columns[~dataset_df.columns.isin(['timestamp', 'label', 'fold'])]][
                dataset_df['fold'] != i + 1][dataset_df['label'].isin([0, 1])].to_numpy()
            y_train = dataset_df['label'][dataset_df['fold'] != i + 1][dataset_df['label'].isin([0, 1])].to_numpy()
            X_test = dataset_df[dataset_df.columns[~dataset_df.columns.isin(['timestamp', 'label', 'fold'])]][
                dataset_df['fold'] == i + 1].to_numpy()
            y_test = dataset_df['label'][dataset_df['fold'] == i + 1].to_numpy()

            clf.fit(X_train, y_train)
            pred_test = clf.predict(X_test)
            results.append(pred_test)

    # Calculate runtime and memory usage
    end_time = time.time()  # End time
    final_memory = memory_usage()[0]  # Final memory usage
    runtime = end_time - start_time
    memory_used = final_memory - initial_memory

    pred_arr = reduce(lambda x, y: np.append(x, y, axis=0), results)
    pred_df = pd.DataFrame(pred_arr, columns=['predicted'])
    result_df = pd.merge(ex_columns, pred_df, left_index=True, right_index=True)

    return result_df, runtime, memory_used

# Command-line argument parsing
argParser = argparse.ArgumentParser()
argParser.add_argument("-f", "--file", help="The CSV file containing dataset.")
argParser.add_argument("-s", "--scenario", help="The scenario to be applied.")
argParser.add_argument("-c", "--classifier", help="The classifier to be used. Can be 'OneClassSVM','MultiClassSVM', 'LOF', 'EE', 'RF', 'KNN'.")
argParser.add_argument("-k", "--kfold", help="The number of folds in the dataset.")
args = argParser.parse_args()

dataset = pd.read_csv(args.file)
scenario = int(args.scenario)
model = args.classifier
k = int(args.kfold)

#These combinations are not appropriate. Please add below if I missed some.
if scenario == 1 and (model == 'MultiClassSVM' or model == 'KNN' or model == 'RF'):
    print("Not appropriate combination of scenario 1 and this classifier.\n")
elif scenario == 2 and (model == 'LOF' or model == 'EE' or model == 'OneClassSVM'):
    print("Not appropriate combination of scenario 2 and this classifier.\n")

else:
    classifier, param_grid = model_selection(model)
    result_df, runtime, memory_used = run_experiment(classifier, param_grid, dataset, scenario, k)

    # Save results to a CSV file
    csv_file_name = f'S{scenario}-{args.classifier}-{args.file}-predicted.csv'
    result_df.to_csv(csv_file_name)

    # Remove the '.csv' extension and append the new suffix for the text file
    txt_file_base = csv_file_name.rsplit('.csv', 1)[0]
    txt_file_name = f'{txt_file_base}-runtime_memory_usage.txt'

    with open(txt_file_name, 'w') as f:
        f.write(f"Runtime: {runtime} seconds\n")
        f.write(f"Memory Used: {memory_used} MiB\n")
