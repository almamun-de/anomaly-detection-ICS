


import pandas as pd
import numpy as np
import argparse

# Constants to define the columns to be excluded from the dataset
EXCLUDE_COLUMNS = ["timestamp", "Timestamp", "attack", "Attack", "label", "Label"]
DISTINCT_VALUES_THRESHOLD = 8  # Threshold for the number of distinct values in Steadytime and Histogram methods


# Function to load data from a CSV file, excluding specific columns like Timestamp and Attack/Label
def load_data(file_path):
    df = pd.read_csv(file_path)
    # Excluding specified columns that are not relevant for the analysis
    return df.drop(columns=EXCLUDE_COLUMNS, errors='ignore')



# MinMax Detection Method: Trains thresholds for each sensor/actuator based on minimum and maximum observed values.
def train_minmax(data):
    thresholds = {}
    # Exclude boolean columns as they are not suitable for the MinMax detection method
    data = data.select_dtypes(exclude=['bool'])

    for col in data.columns:
        min_val = data[col].min()
        max_val = data[col].max()
        # Calculate error margins based on task requirements
        min_err = min_val - (max_val - min_val) / 2
        max_err = max_val + (max_val - min_val) / 2
        thresholds[col] = [min_err, max_err]

        # Print the values for each sensor or actuator
        print(f"Sensor/Actuator: {col}")
        print(f"min_val: {min_val}")
        print(f"max_val: {max_val}")
        print(f"min_err: {min_err}")
        print(f"max_err: {max_err}\n")

    return thresholds



# Function to detect anomalies using the MinMax method
def detect_minmax(data, thresholds):
    alarms = pd.DataFrame(index=data.index)
    alarm_columns = {}  # Dictionary to store alarm columns

    # Iterating over each column and applying the MinMax thresholds
    for col, (min_threshold, max_threshold) in thresholds.items():
        alarm_col = (data[col] < min_threshold) | (data[col] > max_threshold)
        alarm_columns[col] = alarm_col

    # Combining individual alarm columns into a single DataFrame
    alarms = pd.DataFrame(alarm_columns)

    return alarms



# Gradient Detection Method with Thresholds and Error Margins
def train_gradient(data):
    thresholds = {}
    # Exclude boolean columns as they are not suitable for the Gradient detection method
    data = data.select_dtypes(exclude=['bool'])

    for col in data.columns:
        # Convert the column data to float
        column_data = data[col].astype(float)
        # Calculate the minimum and maximum gradients for the current column
        min_gradient = np.min(np.gradient(column_data))
        max_gradient = np.max(np.gradient(column_data))
        # Calculate error margins
        min_err = min_gradient - (max_gradient - min_gradient) / 2
        max_err = max_gradient + (max_gradient - min_gradient) / 2
        thresholds[col] = [min_err, max_err]

        # Print the values for each sensor or actuator
        print(f"Sensor/Actuator: {col}")
        print(f"min_gradient: {min_gradient}")
        print(f"max_gradient: {max_gradient}")
        print(f"min_err: {min_err}")
        print(f"max_err: {max_err}\n")

    return thresholds



# Function to detect anomalies using the Gradient method
def detect_gradient(data, thresholds):
    alarms = pd.DataFrame(index=data.index)
    alarm_columns = {}  # Create a dictionary to store alarm columns

    # Applying the Gradient thresholds to each column to detect anomalies
    for col, (min_threshold, max_threshold) in thresholds.items():
        # Convert the column data to float
        column_data = data[col].astype(float)

        grads = np.gradient(column_data)
        alarm_col = (grads < min_threshold) | (grads > max_threshold)
        alarm_columns[col] = alarm_col

    # Combining individual alarm columns into a single DataFrame
    alarms = pd.DataFrame(alarm_columns)

    return alarms



# Steadytime Detection Method: Trains thresholds based on the duration a sensor/actuator value remains unchanged.
def train_steadytime(data):
    thresholds = {}
    # Convert all data to float for consistent processing
    data = data.astype(float)

    for col in data.columns:
        # Process only if the number of unique values is within the set threshold
        if data[col].nunique() <= DISTINCT_VALUES_THRESHOLD:
            steady_times = []  # List to store durations of steady values
            last_value = data[col].iloc[0]  # Initialize with the first value
            steady_time = 1  # Initialize steady time to count the number of consecutive occurrences of the same value.

            # Loop through each value to calculate steady times
            for value in data[col].iloc[1:]:
                if value == last_value:
                    steady_time += 1
                else:
                    steady_times.append(steady_time)
                    steady_time = 1  # Reset steady time
                    last_value = value

            steady_times.append(steady_time)  # Append the last steady time

            # Determine the minimum and maximum steady times
            min_val, max_val = min(steady_times), max(steady_times)
            # Calculate error margins
            min_err = min_val - (max_val - min_val) / 2
            max_err = max_val + (max_val - min_val) / 2
            thresholds[col] = [min_err, max_err]

            # Print the values for each sensor or actuator
            print(f"Sensor/Actuator: {col}")
            print(f"min_val: {min_val}")
            print(f"max_val: {max_val}")
            print(f"min_err: {min_err}")
            print(f"max_err: {max_err}\n")

    return thresholds



# Function to detect anomalies using the Steadytime method.
def detect_steadytime(data, thresholds):
    # Convert all data to float for consistent processing
    data = data.astype(float)
    alarms = pd.DataFrame(index=data.index)
    alarm_columns = {}  # Create a dictionary to store alarm columns

    # Applying the Steadytime thresholds to each column to detect anomalies
    for col, (min_threshold, max_threshold) in thresholds.items():
        if data[col].nunique() <= DISTINCT_VALUES_THRESHOLD:
            last_value = data[col].iloc[0]
            steady_time = 1
            steady_times = np.array([])
            for value in data[col].iloc[1:]:
                if value == last_value:
                    steady_time += 1
                else:
                    steady_time = 1
                    last_value = value
                steady_times = np.append(steady_times, steady_time)
                alarm_col = (steady_times < min_threshold) | (steady_times > max_threshold)
                alarm_columns[col] = alarm_col

    alarms = pd.DataFrame(alarm_columns)  # Combining individual alarm columns into a single DataFrame
    return alarms



# Histogram Detection Method: Trains thresholds based on the distribution of sensor/actuator values in a fixed window.
def train_histogram(data, window_size):
    thresholds = {}
    # Convert all data to float for consistent processing
    data = data.astype(float)

    for col in data.columns:
        # Process only columns with a small number of distinct values
        if data[col].nunique() <= DISTINCT_VALUES_THRESHOLD:  # distinct values (<= 8)
            # Determine the bin edges (including both ends)
            unique_vals = sorted(data[col].unique())
            bin_edges = np.linspace(unique_vals[0], unique_vals[-1], len(unique_vals) + 1)

            # Manually compute histogram
            hist = [((data[col] >= bin_edges[i]) & (data[col] < bin_edges[i + 1])).sum() for i in range(len(unique_vals))]

            # Calculate the frequency of occurrences within the window size
            hist_freq = np.array(hist) / window_size

            # Determine min and max frequencies for error margin calculation
            min_val, max_val = hist_freq.min(), hist_freq.max()
            # Calculate error margins
            min_err = min_val - (max_val - min_val) / 2
            max_err = max_val + (max_val - min_val) / 2
            thresholds[col] = [min_err, max_err]

            # Print the values for each sensor or actuator
            print(f"Sensor/Actuator: {col}")
            print(f"min_val: {min_val}")
            print(f"max_val: {max_val}")
            print(f"min_err: {min_err}")
            print(f"max_err: {max_err}\n")

    return thresholds



# Function to detect anomalies based on histogram comparisons
def detect_histogram(data, thresholds, window_size):
    # Convert all data to float for consistent processing
    data = data.astype(float)
    alarms = pd.DataFrame(index=data.index)
    alarm_columns = []  # List to store alarm columns

    for col, (min_threshold, max_threshold) in thresholds.items():
        # Calculate the bin edges (including both ends)
        unique_vals = sorted(data[col].unique())
        bin_edges = np.linspace(unique_vals[0], unique_vals[-1], len(unique_vals) + 1)

        # Manually compute histogram
        hist = [((data[col] >= bin_edges[i]) & (data[col] < bin_edges[i + 1])).sum() for i in range(len(unique_vals))]

        # Calculate the frequency of occurrences within the window size
        hist_freq = np.array(hist) / window_size
        min_val, max_val = hist_freq.min(), hist_freq.max()

        # Determine if the histogram frequency falls outside the trained thresholds
        alarm_col = (min_val < min_threshold) | (max_val > max_threshold)
        # Append the alarm data and column name to the list
        alarm_columns.append((col, alarm_col))

    # Convert the list of tuples into a DataFrame for easier analysis and visualization
    alarms = pd.DataFrame(alarm_columns)

    return alarms




# Main function to control the flow of the program based on user input
def main(train_file, test_file, method):
    train_data = load_data(train_file)
    test_data = load_data(test_file)
    window_size_train = len(train_data)
    window_size_test = len(test_data)

    if method == 'minmax':
        thresholds = train_minmax(train_data)
        alarms = detect_minmax(test_data, thresholds)
    elif method == 'gradient':
        thresholds = train_gradient(train_data)
        alarms = detect_gradient(test_data, thresholds)
    elif method == 'steadytime':
        thresholds = train_steadytime(train_data)
        alarms = detect_steadytime(test_data, thresholds)
    elif method == 'histogram':
        thresholds = train_histogram(train_data, window_size_train)
        alarms = detect_histogram(test_data, thresholds, window_size_test)
    else:
        raise ValueError("Invalid method. Choose from minmax, gradient, steadytime, histogram.")


    # Save the results
    alarms.to_csv(f"anomaly_detection_results_by_{args.method}.csv")
    print(f"Anomaly detection completed. Results saved to 'anomaly_detection_results_by_{args.method}.csv'.")


if __name__ == "__main__":
    # Parser setup for command line arguments and triggering the main function
    parser = argparse.ArgumentParser(description='Anomaly Detection in Industrial Systems')
    parser.add_argument('train_file', type=str, help='Path to the training dataset CSV file')
    parser.add_argument('test_file', type=str, help='Path to the testing dataset CSV file')
    parser.add_argument('method', choices=['minmax', 'gradient', 'steadytime', 'histogram'], help='Anomaly detection method')
    args = parser.parse_args()

    main(args.train_file, args.test_file, args.method)
