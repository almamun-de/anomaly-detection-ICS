#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from scapy.all import *
from functools import reduce
from sklearn import preprocessing
import keras
import tensorflow as tf
import argparse


#Using aotoencoder to extract feature from dataset while applying the scenario given.
def extract_feature(autoencoder, encoder, dataset_df, scenario, k):
    ex_columns_arr = dataset_df[dataset_df.columns[dataset_df.columns.isin(['timestamp','label','fold'])]].sort_values(['fold','timestamp']).to_numpy()
    ex_columns = pd.DataFrame(ex_columns_arr,columns=['timestamp','label','fold'])
    feature_lst=[]
    min_max_scaler = preprocessing.MinMaxScaler()
    
    if scenario == 1:
        for i in range(k):
            X_train = dataset_df[dataset_df.columns[~dataset_df.columns.isin(['timestamp','label','fold','Unnamed: 0'])]][dataset_df['fold']!=i+1][dataset_df['label']==0].to_numpy()
            y_train = dataset_df['label'][dataset_df['fold']!=i+1][ dataset_df['label']==0].to_numpy()
            X_test = dataset_df[dataset_df.columns[~dataset_df.columns.isin(['timestamp','label','fold','Unnamed: 0'])]][dataset_df['fold']==i+1].to_numpy()
            y_test = dataset_df['label'][dataset_df['fold']==i+1].to_numpy()
            
            X_train = min_max_scaler.fit_transform(X_train)
            X_test = min_max_scaler.transform(X_test)

            autoencoder.fit(X_train,X_train, epochs=1, validation_split=0.2)
            feature = encoder.predict(X_test)
            feature_lst.append(feature)
            
    elif scenario == 2:
        for i in range(k):
            X_train = dataset_df[dataset_df.columns[~dataset_df.columns.isin(['timestamp','label','fold','Unnamed: 0'])]][dataset_df['fold']!=i+1][dataset_df['label']!=1].to_numpy()
            y_train = dataset_df['label'][dataset_df['fold']!=i+1][dataset_df['label']!=1].to_numpy()
            X_test = dataset_df[dataset_df.columns[~dataset_df.columns.isin(['timestamp','label','fold','Unnamed: 0'])]][dataset_df['fold']==i+1].to_numpy()
            y_test = dataset_df['label'][dataset_df['fold']==i+1].to_numpy()
            
            X_train = min_max_scaler.fit_transform(X_train)
            X_test = min_max_scaler.transform(X_test)

            autoencoder.fit(X_train,X_train, epochs=1, validation_split=0.2)
            feature = encoder.predict(X_test)
            feature_lst.append(feature)
    
    elif scenario == 3:
        for i in range(k):
            X_train = dataset_df[dataset_df.columns[~dataset_df.columns.isin(['timestamp','label','fold','Unnamed: 0'])]][dataset_df['fold']!=i+1][dataset_df['label'].isin([0,1])].to_numpy()
            y_train = dataset_df['label'][dataset_df['fold']!=i+1][dataset_df['label'].isin([0,1])].to_numpy()
            X_test = dataset_df[dataset_df.columns[~dataset_df.columns.isin(['timestamp','label','fold','Unnamed: 0'])]][dataset_df['fold']==i+1].to_numpy()
            y_test = dataset_df['label'][dataset_df['fold']==i+1].to_numpy()
            
            X_train = min_max_scaler.fit_transform(X_train)
            X_test = min_max_scaler.transform(X_test)

            autoencoder.fit(X_train,X_train, epochs=1, validation_split=0.2)
            feature = encoder.predict(X_test)
            feature_lst.append(feature)
    
    #create results dataframe
    feature_arr = reduce(lambda x, y: np.append(x, y, axis=0), feature_lst)
    columns_lst=[]
    for i in range(len(feature_arr[0])):
        columns_lst.append('feature'+str(i+1))
    feature_df = pd.DataFrame(feature_arr, columns=columns_lst)
    result_df = pd.merge(ex_columns,feature_df,left_index=True,right_index=True)
    
    return result_df        


# Create packet features from the dataset. They can be used as input for autoencoder in order to extract feature from network packets.

def create_packet_df(dataset):
    rawpkt = b'\x00\x00\xbc\xd1`\xdax\xe7\xd1\xe0\x02^\x08\x00E\x00\x00zp&@\x00\x80\x06\x00\x00\x8dQ\x00\n\x8dQ\x00S\xc4c\xaf\x12\xdd\x88\x8d\x87\x94\x95CQP\x18\xf9t\x1bl\x00\x00p\x00:\x00\x00\x01\x02\x10\x00\x00\x00\x00\x1a9/\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\n\x00\x02\x00\xa1\x00\x04\x00\t\x135\x00\xb1\x00&\x00\xe4j\n\x02 \x02$\x01\x02\x00\x06\x00\x12\x00L\x02 r$\x00\x00\xce\x04\x00\x01\x00L\x02 r$\x00,=\x04\x00\x01\x00'
    ex_columns = dataset[dataset.columns[dataset.columns.isin(['timestamp','label','fold'])]]
    data_df = dataset[dataset.columns[~dataset.columns.isin(['timestamp','label','fold'])]]
    pkt_df_lst=[]
    for col in data_df.columns:
        pkt_lst=[]
        lst = data_df[col].to_list()
        for item in lst:
            load = raw(struct.pack("f", item))
            pkt = list(rawpkt+load)
            pkt_lst.append(pkt)
        columns_lst=[]
        for i in range(len(pkt_lst[0])):
            columns_lst.append('byte'+str(i+1))
        pkt_df = pd.DataFrame(pkt_lst,columns=columns_lst)
        new_pkt_df = pd.merge(ex_columns,pkt_df,left_index=True,right_index=True)
        new_pkt_df.to_csv(col+'.csv')
        print("Created file "+col+'.csv\n')
        
    #result_df = pd.concat(pkt_df_lst,ignore_index=True)
    
    return 0



argParser = argparse.ArgumentParser()
argParser.add_argument("-f", "--file", help="The file containing original data and k-fold information")
argParser.add_argument("-m", "--mode", help="The running mode of the script. Can be 'feature' or 'packet'.")
argParser.add_argument("-s", "--scenario", help="The scenario to be applied.")
argParser.add_argument("-k", "--kfold", help="The unmber of folds in the original dataset.")
args = argParser.parse_args()

dataset=pd.read_csv(args.file)

#dimension of data. Ignore 'timestamp','label','fold','Unnamed: 0' so subtracted by 4.
n=len(dataset.columns)-4

#Build model 1
input_data_1 = keras.Input(shape=(n,))
encoded_1 = keras.layers.Dense(64, activation='relu')(input_data_1)
encoded_1 = keras.layers.Dense(32, activation='relu')(encoded_1)
encoded_1 = keras.layers.Dense(16)(encoded_1)

decoded_1 = keras.layers.Dense(32, activation='relu')(encoded_1)
decoded_1 = keras.layers.Dense(64, activation='relu')(decoded_1)
decoded_1 = keras.layers.Dense(n)(decoded_1)

autoencoder_1 = keras.Model(input_data_1,decoded_1)
autoencoder_1.compile(optimizer='adam', loss='mean_squared_error')
encoder_1 = keras.Model(input_data_1, encoded_1)


autoencoder = autoencoder_1
encoder = encoder_1


if args.mode == 'feature':
    feature_df = extract_feature(autoencoder, encoder, dataset, scenario=int(args.scenario), k=int(args.kfold))
    feature_df.to_csv('S'+str(args.scenario)+'-'+str(args.file)+'-feature.csv')
    print("Created feature file\n")

elif args.mode == 'packet':

    create_packet_df(dataset)
