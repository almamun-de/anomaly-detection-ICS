Usage:
The script takes two positional arguments: the path to the training CSV file and the path to the testing CSV file.

To run the script, use the following command:
python pca.py path_to_train_csv path_to_test_csv

[Replace path_to_train_csv and path_to_test_csv with the actual file paths of your datasets.]

Output:
The script will generate PCA-transformed CSV files for each β value specified in the code. The files will be named in the format X_originalfilename_pca_beta.csv.


----------------------------------
# PCA Feature Extraction

This Python script performs Principal Component Analysis (PCA) on provided training and testing datasets to reduce feature dimensions while retaining a significant portion of the variance. It is tailored for datasets with sensor readings and actuator states, excluding specific columns like timestamps and attack/labels.

