import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from venn import venn
from matplotlib.backends.backend_pdf import PdfPages

def create_sets(df_lst):
    """
    Creates sets of indices for classification errors for each classifier.
    Returns:
    tuple: Two lists of sets.
           The first list contains sets of indices where predictions were incorrect.
           The second list contains sets of tuples (index, predicted_label) for incorrect predictions.
    """
    sets_1 = []  # sets for task 3(b)
    sets_2 = []  # sets for task 3(c)

    for df in df_lst:
        # Unsupervised classifiers 'predicted' == -1 means attack predicted.
        if -1 in df['predicted'].to_numpy():
            # Convert all non-zero labels to 1
            df.loc[df['label'] != 0, 'label'] = 1
            # Convert predicted labels of -1 to 1. Because for Unsupervised model, predicted == -1 means attack predicted.
            df.loc[df['predicted'] == 1, 'predicted'] = 0
            # Convert predicted labels of 1 to 0. Because for Unsupervised model, predicted == 1 means no attack predicted.
            df.loc[df['predicted'] == -1, 'predicted'] = 1

        # Extracting indices where predictions were incorrect.
        idx = df['Unnamed: 0'][df['label'] != df['predicted']].to_numpy()
        # Extracting predicted labels for incorrect predictions.
        pred = df['predicted'][df['label'] != df['predicted']].to_numpy()
        # Creating a 2D array of [index, predicted_label].
        idx_pred = np.column_stack([idx, pred])

        set1 = set(idx)
        set2 = set(map(tuple, idx_pred))

        sets_1.append(set1)
        sets_2.append(set2)

    return sets_1, sets_2

def venn_diagram(sets1, sets2, classifiers, filename):
    """
    Generates and saves Venn diagrams for classification errors in a PDF.
    """
    # Mapping classifiers to their respective sets.
    data1 = dict(zip(classifiers, sets1))
    data2 = dict(zip(classifiers, sets2))

    with PdfPages(filename) as pdf:
        # Plotting Venn diagram for task 3(b).
        plt.figure()
        venn(data1)
        plt.title('Task 3(b): Incorrectly classified by each classifier.')
        pdf.savefig()
        plt.close()

        # Plotting Venn diagram for task 3(c).
        plt.figure()
        venn(data2)
        plt.title('Task 3(c): Incorrectly classified with the same wrong label by each classifier.')
        pdf.savefig()
        plt.close()


# Parsing command-line arguments.
argParser = argparse.ArgumentParser()
argParser.add_argument("-f", "--files", nargs='*', help="CSV files with original label and predicted results.")
argParser.add_argument("-s", "--scenario", type=int, help="Scenario number applied to the results.")
argParser.add_argument("-c", "--classifier", nargs='*', help="Classifiers used, in the same order as input files.")
args = argParser.parse_args()

# Reading data from input CSV files.
df_lst = [pd.read_csv(item) for item in args.files]

# Extracting scenario and classifier information.
scenario = args.scenario
classifiers = args.classifier

# Creating sets for Venn diagrams.
sets1, sets2 = create_sets(df_lst)

# Generating filename for the output PDF.
filename = f'S{scenario}-{"-".join(classifiers)}.pdf'

# Creating and saving Venn diagrams.
venn_diagram(sets1, sets2, classifiers, filename)

# Example usage:
# python plot_venn.py -f file1.csv file2.csv file3.csv -s 1 -c classifier1 classifier2 classifier3
# list of classifiers are expected to be in the same order as the CSV files.
