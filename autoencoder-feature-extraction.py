
# Function to extract features from a dataset using an autoencoder.
# It applies different scenarios for data filtering and uses k-fold cross-validation.
def extract_feature(autoencoder, encoder, dataset_df, scenario, k):
    # Exclude columns not used for feature extraction
    ex_columns = dataset_df[dataset_df.columns[dataset_df.columns.isin(['timestamp', 'label', 'fold'])]]
    feature_lst = []
    min_max_scaler = preprocessing.MinMaxScaler()

    # Loop through each fold for cross-validation
    for i in range(k):
        # Apply different scenarios for data filtering
        if scenario == 1:
            # Scenario 1: Exclude current fold and label 0
            condition = (dataset_df['fold'] != i + 1) & (dataset_df['label'] == 0)
        elif scenario == 2:
            # Scenario 2: Exclude current fold and label 1
            condition = (dataset_df['fold'] != i + 1) & (dataset_df['label'] != 1)
        elif scenario == 3:
            # Scenario 3: Exclude current fold and include labels 0 and 1
            condition = (dataset_df['fold'] != i + 1) & (dataset_df['label'].isin([0, 1]))

        # Prepare training and testing data
        X_train = dataset_df[dataset_df.columns[~dataset_df.columns.isin(['timestamp', 'label', 'fold', 'Unnamed: 0'])]][condition].to_numpy()
        X_test = dataset_df[dataset_df.columns[~dataset_df.columns.isin(['timestamp', 'label', 'fold', 'Unnamed: 0'])]][dataset_df['fold'] == i + 1].to_numpy()

        # Normalize data
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.transform(X_test)

        # Train the autoencoder
        autoencoder.fit(X_train, X_train, epochs=1, validation_split=0.2)

        # Extract features using the encoder
        feature = encoder.predict(X_test)
        feature_lst.append(feature)

    # Combine features from all folds
    feature_arr = np.concatenate(feature_lst, axis=0)
    columns_lst = ['feature' + str(i + 1) for i in range(feature_arr.shape[1])]
    feature_df = pd.DataFrame(feature_arr, columns=columns_lst)
    result_df = pd.merge(ex_columns, feature_df, left_index=True, right_index=True)

    return result_df

# Main function for executing the script with command line arguments.
# It includes memory and runtime profiling.
def main():
    # Setup argument parser
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-f", "--file", help="The file containing original data and k-fold information")
    argParser.add_argument("-m", "--mode", help="The running mode of the script. Can be 'feature' or 'packet'.")
    argParser.add_argument("-s", "--scenario", help="The scenario to be applied.")
    argParser.add_argument("-k", "--kfold", help="The number of folds in the original dataset.")
    args = argParser.parse_args()

    # Load dataset from the specified file
    dataset = pd.read_csv(args.file)

    # Determine the dimension of the data (excluding specific columns)
    n = len(dataset.columns) - 4

    # Building the autoencoder model
    input_data = keras.Input(shape=(n,))
    encoded = keras.layers.Dense(64, activation='relu')(input_data)
    encoded = keras.layers.Dense(32, activation='relu')(encoded)
    encoded = keras.layers.Dense(16)(encoded)

    decoded = keras.layers.Dense(32, activation='relu')(encoded)
    decoded = keras.layers.Dense(64, activation='relu')(decoded)
    decoded = keras.layers.Dense(n)(decoded)

    autoencoder = keras.Model(input_data, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    encoder = keras.Model(input_data, encoded)

    # Feature extraction mode
    if args.mode == 'feature':
        # Start profiling for runtime and memory
        start_time = time.time()  # Start timer
        initial_memory = memory_usage()[0]  # Initial memory usage

        # Extract features based on the given scenario and k-fold
        feature_df = extract_feature(autoencoder, encoder, dataset, scenario=int(args.scenario), k=int(args.kfold))

        # End profiling
        end_time = time.time()  # End timer
        final_memory = memory_usage()[0]  # Final memory usage

        # Calculate and print runtime and memory usage
        elapsed_time = end_time - start_time
        memory_used = final_memory - initial_memory
        output_str = f"Autoencoder Feature Extraction\nRuntime: {elapsed_time} seconds\nMemory Used: {memory_used} MiB\n"
        print(output_str)

        # Save extracted features to CSV
        csv_file_name = f'S{args.scenario}-{args.file}-feature.csv'
        feature_df.to_csv(csv_file_name)
        print("Created feature file\n")

        # Save performance metrics to a text file
        txt_file_base = csv_file_name.rsplit('.csv', 1)[0]
        txt_file_name = f'{txt_file_base}-Autoencoder_Features_Performance.txt'
        with open(txt_file_name, 'w') as file:
            file.write(output_str)

# Execute the main function if the script is run as the main program
if __name__ == "__main__":
    main()

# Example usage command:
'''
python autoencoder-feature-extraction-runtime-memory -f your_data.csv -m feature -s 1 -k 10
'''
