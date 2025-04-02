#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from functools import reduce
import argparse

#Re-create the label for test files. New label indicates the type of attack of an instance of data.
def test_relabeling(test_df, dur_lst, atk_lst):
    for duration ,label in zip(dur_lst,atk_lst):
        if(label == 1):
            idx = test_df[test_df['label']==1][0:duration+1].index
            test_df.loc[idx, 'label'] = 'x'
        else:
            idx = test_df[test_df['label']==1][0:duration+1].index
            test_df.loc[idx, 'label'] = label            
    
    test_df.loc[test_df['label']=='x','label'] = 1
    
    return test_df
        
#Create label for train files. All labels are set 0, since there is no attack in training data.
def train_labeling(train_df):
    n = len(train_df)
    label_arr = np.zeros(n, dtype=int)
    train_df.insert(len(train_df.columns),'label',label_arr)
    
    return train_df

#Apply the k-fold algorithm to a dataset.  The 'fold' column created here indicates to which fold the data belongs

def create_fold_label(merged_df, k):
    n = len(merged_df)
    fold_arr = np.full(n, 'x')
    merged_df.insert(len(merged_df.columns),'fold',fold_arr)
    for atk in range(53):
        length = int(len(merged_df['label'][merged_df['label']== atk])/k)
        for i in range(k-1):
            idx = merged_df.loc[(merged_df['fold']=='x') & (merged_df['label']==atk)][0:length].index
            merged_df.loc[idx,'fold'] = i+1
        idx = merged_df.loc[(merged_df['fold']=='x') & (merged_df['label']==atk)].index
        merged_df.loc[idx,'fold'] = k
    
    return merged_df

argParser = argparse.ArgumentParser()
argParser.add_argument("-k", "--kfold", help="The number of folds to be used")
args = argParser.parse_args()

k = int(args.kfold)

train_files = ['hai-train1.csv','hai-train2.csv','hai-train3.csv','hai-train4.csv']
test_files = ['hai-test1.csv','hai-test2.csv']
label_files = ['label-test1.csv','label-test2.csv']

labeled_df_lst = []

for train in train_files:   
    train_df = pd.read_csv(train)
    
    labeled = train_labeling(train_df)
    
    labeled_df_lst.append(labeled)

#Attack duration information from file 'hai-techenical-details.pdf'.
duration_lst1=[237,198,156,164,161,197,604,96,130,55,131,78,133,627]
duration_lst2=[132,131,68,122,85,196,614,133,85,88,204,127,539,61,147,95,505,214,131,131,82,211,79,107,60,118,132,155,115,154,95,153,2051,529,86,119,189,122]
duration_lsts=[duration_lst1,duration_lst2]

#Corresponding attack types
atk_lst1=list(np.arange(1,1+len(duration_lst1)))
atk_lst2=list(np.arange(15,15+len(duration_lst2)))
atk_lsts=[atk_lst1,atk_lst2]

#Apply the k-fold algorithm. Data of each tyoe of attack are evenly distributed into each fold.
for test, label, dur_lst, atk_lst in zip(test_files,label_files, duration_lsts, atk_lsts):
    test_df = pd.read_csv(test)
    label_df = pd.read_csv(label)
    label_df = label_df[label_df.columns[label_df.columns!='timestamp']]
    new_test_df = test_df.merge(label_df, left_index=True, right_index=True)
    
    labeled = test_relabeling(new_test_df, dur_lst, atk_lst)    
    labeled_df_lst.append(labeled)

result_df = pd.concat(labeled_df_lst,ignore_index=True)

merged_df = create_fold_label(result_df, k)

merged_df.to_csv(str(k)+'-fold-data.csv')
