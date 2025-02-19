import pandas as pd


def merge_files(hai_filename, label_filename, output_filename):
    # Reading the CSV files
    hai_df = pd.read_csv(hai_filename)
    label_df = pd.read_csv(label_filename)

    # Adjusting the timestamp format of the label dataframe if needed
    label_df['timestamp'] = pd.to_datetime(label_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

    # Determine the correct column name for merging
    merge_column = 'timestamp' if 'timestamp' in hai_df.columns else 'Timestamp'

    # Merging the two dataframes based on the determined merge column
    merged_df = pd.merge(hai_df, label_df, on=merge_column, how='inner')

    # Rearranging the columns to ensure the merge column is the first column
    cols = [merge_column] + [col for col in merged_df if col != merge_column]
    merged_df = merged_df[cols]

    # Saving the merged dataframe to the specified output filename
    merged_df.to_csv(output_filename, index=False)
    '''The index=False argument ensures that the DataFrame's index is not written to the CSV.'''



# Filenames for the first pair
hai_test1_filename = 'hai-test1.csv'
label_test1_filename = 'label-test1.csv'
output1_filename = 'hai-test1-label.csv'

# Filenames for the second pair
hai_test2_filename = 'hai-test2.csv'
label_test2_filename = 'label-test2.csv'
output2_filename = 'hai-test2-label.csv'

# Merging the files
merge_files(hai_test1_filename, label_test1_filename, output1_filename)
merge_files(hai_test2_filename, label_test2_filename, output2_filename)