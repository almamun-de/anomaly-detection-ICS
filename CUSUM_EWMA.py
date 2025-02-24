
import pandas as pd
from functools import reduce
import numpy as np
import argparse

# Traning procecss of CUSUM

def CUSUM_train(df):

    lst = []
    for col in df.columns:
        s = 0.0
        for i in df[col].index:
            if i == 0:
                continue
            else:
                rk = abs(df[col][i]-df[col][i-1])
                s += rk
        # calculate the average of rks and set beta accordingly s.t. expected value of (rk - beta) < 0 
        beta = s/(len(df[col].index)-1)
        # store the results in a list
        lst.append([col, beta])
    
    # create a new dataframe containing results    
    result_df = pd.DataFrame(lst, columns=['sensor', 'beta'])
    
    return result_df
    


# Detection process of CUSUM
# Input parameter 'betas' must be the dataframe generated from training process or one with the same layout
def CUSUM_detect(df, betas, threshold):
    df_lst=[]
    for col in df.columns:
        s = 0.0
        beta = betas['beta'][betas['sensor'] == col].iloc[0]
        alarm_lst = []
        for i in df[col].index:
            if i == 0:
                s = 0.0
            else:
                # applying the recursive definition of CUSUM
                rk = abs(df[col][i]-df[col][i-1])
                s += (rk-beta)
                s = max(0.0,s)
            # compare S with threshold and store results. 0 as normal and 1 as alarm.
            if s > threshold:
                alarm_lst.append(1)
                s = 0.0
            else:
                alarm_lst.append(0)
        # create dataframe containing result for each columns and store in a list.        
        col_df = pd.DataFrame(alarm_lst, columns=[col])
        df_lst.append(col_df)
        
    # merge the results of each column to get the final result.   
    result_df = reduce(lambda x, y: pd.merge(x, y,left_index=True, right_index=True), df_lst)    
    return result_df




# Training process of EWMA
def EWMA_train(df, threshold):
    lst=[]
    # The possible values to be chosen for alpha during training
    # Recommended value between 0.05 to 0.25, and here we choose 5 values in the recommended interval
    cand_alpha = [0.05,0.10,0.15,0.20,0.25]
    for col in df.columns:
        min_count = len(df[col].index)
        opt_alpha = 0.05  # default alpha value
        for alpha in cand_alpha:
            # execute EWMA with each possible alpha and choose the one gives fewest false alarms.
            count = 0 # counter for false alarms.
            for i in df[col].index:
                if i == 0:
                    s = 0.0
                else:
                    s = alpha*df[col][i] + (1-alpha)*s
                    
                if s > threshold:
                    count += 1
                    s = 0.0
            if count < min_count:
                min_count = count
                opt_alpha = alpha
        lst.append([col,opt_alpha]) # list containing results.
    # create dataframe with the results.
    result_df = pd.DataFrame(lst, columns=['sensor','alpha'])
    
    return result_df



# Detection process of EWMA
# parameter 'alphas' munst be the dataframe from the training process or one of the same layout.
def EWMA_detect(df, alphas, threshold):
    df_lst=[]
    for col in df.columns:
        s = 0.0
        alpha = alphas['alpha'][alphas['sensor']==col].iloc[0]
        alarm_lst = []
        for i in df[col].index:
            if i == 0:
                s = 0.0
            else:
                # applying the recursive definition of EWMA
                s = alpha*df[col][i] + (1-alpha)*s
            # create alarms if S > threshold
            if s > threshold:
                alarm_lst.append(1)
                s = 0.0
            else:
                alarm_lst.append(0)
        # result for each column is stored in to list        
        col_df = pd.DataFrame(alarm_lst, columns=[col])
        df_lst.append(col_df)
    # merge the results of all columns and get final result.
    result_df = reduce(lambda x, y: pd.merge(x, y,left_index=True, right_index=True), df_lst)        
    
    return result_df




argParser = argparse.ArgumentParser()
argParser.add_argument("-f", "--file",nargs=2 , help="The csv files to be read. Train and test data need to be entered in order.")
argParser.add_argument("-m", "--method", help="The algorithm to be used. Enter either 'CUSUM' or 'EWMA'.")
argParser.add_argument("-t", "--threshold", help="The threshold to be used. Should be float ponit number.")
args = argParser.parse_args()


# Read csv file of training and testing data.
train_df = pd.read_csv(args.file[0])
test_df = pd.read_csv(args.file[1])

EXCLUDE_COLUMNS = ["timestamp", "Timestamp", "attack", "Attack", "label", "Label"]

# Filtering out the specified columns from train_df
train_df = train_df[train_df.columns[~train_df.columns.isin(EXCLUDE_COLUMNS)]]

# Filtering out the specified columns from test_df
test_df = test_df[test_df.columns[~test_df.columns.isin(EXCLUDE_COLUMNS)]]


# Make sure threshold is float type.
threshold = float(args.threshold)

# excute diffrent methods according to user input.
# the parameters trained from training data are stored into a seperate csv file.
if args.method == 'CUSUM':
    beta_df = CUSUM_train(train_df)
    alarm_df = CUSUM_detect(test_df, beta_df, threshold)
    beta_df.to_csv('CUSUM-betas.csv')
    
elif args.method == 'EWMA':
    alpha_df = EWMA_train(train_df, threshold)
    alarm_df = EWMA_detect(test_df, alpha_df, threshold)
    alpha_df.to_csv('EWMA-alphas.csv')

# the results containing alarms for testing data are stored into a csv file.    
filename = args.method +'-'+ args.file[0] +'-'+ args.file[1] +'.csv'
    
alarm_df.to_csv(filename)    
