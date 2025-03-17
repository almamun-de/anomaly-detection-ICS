import pandas as pd


def merge_files(end_filename, label_filename, output_filename):
    # Reading the CSV files
    end_df = pd.read_csv(end_filename)
    label_df = pd.read_csv(label_filename)

    # Determine the correct column name for merging in each dataframe
    merge_column_end = 'timestamp' if 'timestamp' in end_df.columns else 'Timestamp'
    merge_column_label = 'timestamp' if 'timestamp' in label_df.columns else 'Timestamp'

    # Adjusting the timestamp format of both dataframes to match
    end_df[merge_column_end] = pd.to_datetime(end_df[merge_column_end]).dt.strftime('%Y-%m-%d %H:%M:%S')
    label_df[merge_column_label] = pd.to_datetime(label_df[merge_column_label]).dt.strftime('%Y-%m-%d %H:%M:%S')

    # Merging the two dataframes based on the determined merge columns using an "inner" merge
    merged_df = pd.merge(end_df, label_df, left_on=merge_column_end, right_on=merge_column_label, how='inner')

    # Drop the extra timestamp column if they are different
    if merge_column_end != merge_column_label:
        merged_df.drop(columns=[merge_column_label], inplace=True)

    # Rearranging the columns to ensure the merge column is the first column
    cols = [merge_column_end] + [col for col in merged_df if col != merge_column_end]
    merged_df = merged_df[cols]

    # Saving the merged dataframe to the specified output filename
    merged_df.to_csv(output_filename, index=False)
    '''The index=False argument ensures that the DataFrame's index is not written to the CSV.'''


output2_filename = 'end-test2-label.csv'

# Merging the files
merge_files(end_test1_filename, label_test1_filename, output1_filename)
merge_files(end_test2_filename, label_test2_filename, output2_filename)
