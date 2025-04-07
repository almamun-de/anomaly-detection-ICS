'''
python pca.py train.csv_file test.csv_file
'''

import pandas as pd
import numpy as np
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='PCA Feature Reduction')
parser.add_argument('train_csv', type=str, help='Path to the training CSV file')
parser.add_argument('test_csv', type=str, help='Path to the testing CSV file')
args = parser.parse_args()

# Define columns to be excluded
EXCLUDE_COLUMNS = ["timestamp", "Timestamp", "attack", "Attack", "label", "Label"]

# Load the datasets
train_data = pd.read_csv(args.train_csv)
test_data = pd.read_csv(args.test_csv)

# Drop the excluded columns
train_data = train_data.drop(columns=EXCLUDE_COLUMNS, errors='ignore')
test_data = test_data.drop(columns=EXCLUDE_COLUMNS, errors='ignore')

# Convert to numpy arrays
X_train = train_data.values
X_test = test_data.values

# Standardize the data
def standardize_data(X):
    mean_vector = np.mean(X, axis=0)    # calculates the mean (or average) of each column
    std_vector = np.std(X, axis=0)  # calculates the standard deviation of each column
    # formula: square root of ( ∑ ( Xi – ų ) ^ 2 ) / N

    # Avoid division by zero for features with zero standard deviation
    std_vector[std_vector == 0] = 1
    return (X - mean_vector) / std_vector


X_train_std = standardize_data(X_train)
X_test_std = standardize_data(X_test)


# Calculate the covariance matrix
def calculate_covariance_matrix(X):
    n_samples = X.shape[0]
    mean_vector = np.mean(X, axis=0)
    cov_matrix = (X - mean_vector).T.dot(X - mean_vector) / (n_samples - 1)
    return cov_matrix  # Square matrix


cov_matrix = calculate_covariance_matrix(X_train_std)

# Calculate eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort the eigenvectors by decreasing eigenvalues
eigenpairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]
eigenpairs.sort(key=lambda x: x[0], reverse=True)

# Separate eigenvalues and eigenvectors
sorted_eigenvalues, sorted_eigenvectors = zip(*eigenpairs)
sorted_eigenvectors = np.array(sorted_eigenvectors).T


# Calculate the explained variance
explained_variances = sorted_eigenvalues / np.sum(sorted_eigenvalues)

# Function to select the number of principal components for each β
def select_k_components(variances, beta):
    total_variance = 0
    for k, variance in enumerate(variances):
        total_variance += variance
        if total_variance > beta:
            return k + 1
    return len(variances)


# Function to transform data according to the selected number of components
def transform_data(X, eigenvectors, k):
    W = np.hstack([eigenvectors[:, i].reshape(-1, 1) for i in range(k)])
    return X.dot(W)


# β values
beta_values = [0.998, 0.919, 0.891]

# Apply PCA for each β value and save the transformed datasets
for beta in beta_values:
    k = select_k_components(explained_variances, beta)
    X_train_pca = transform_data(X_train_std, sorted_eigenvectors, k)
    X_test_pca = transform_data(X_test_std, sorted_eigenvectors, k)
    np.savetxt(f'X_{args.train_csv}_pca_{beta}.csv', X_train_pca, delimiter=',')
    np.savetxt(f'X_{args.test_csv}_pca_{beta}.csv', X_test_pca, delimiter=',')
