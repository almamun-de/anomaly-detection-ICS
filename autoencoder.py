#!/usr/bin/env python
# coding: utf-8

import argparse
import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing
from scapy.all import *
import matplotlib.pyplot as plt


# The function to create lists of True Positive Rate and False Positive Rate under different thrsholds. The lists will be used to plot ROC Curve.
def generate_roc_prameters(test, reconstructed, label, threshold_lst):
    fpr_lst=[]
    tpr_lst=[]
    
    for threshold in threshold_lst:
        predicted = classifier_predict(test, reconstructed, threshold)
        _,_,_,tpr,fpr = metric_calculation(label, predicted)
        tpr_lst.append(tpr)
        fpr_lst.append(fpr)
        
    return tpr_lst, fpr_lst     

# Classifier function for 2 vectors
def classifier(v1, v2, threshold):
    count = 0.0
    for dim1, dim2 in zip(v1, v2):
        count += (dim1 - dim2)**2
        
    count /= len(v1)
    if count > threshold:
        return 1
    else:
        return 0
    
#Function to calculate metrics of accuracy, precision, recall, TPR and FPR.
def metric_calculation(label, predicted):
    true_positive = 0
    true_negetive = 0
    false_positive = 0
    false_negetive = 0
    for s1, s2 in zip(label, predicted):
        if (s1 == 1) and (s1 == s2):
            true_positive += 1
        elif (s1 == 0) and (s1 == s2):
            true_negetive += 1
        elif (s1 == 1) and (s1 != s2):
            false_negetive += 1
        elif (s1 == 0) and (s1 != s2):
            false_positive += 1
    
    accuracy = (true_positive + true_negetive)/(true_negetive + true_positive + false_negetive + false_positive)
    
    # If threshold is chosen inappropriately, there is need to deal with division by zero error.
    if(true_positive + false_positive)==0:
        precision = 'DivisionByZero'    
    else:
        precision = true_positive/(true_positive + false_positive)                
    if(true_positive + false_negetive)==0:
        recall = 'DivisionByZero'    
    else:
        recall = true_positive/(true_positive + false_negetive)        
    tpr = recall
    if(false_positive + true_negetive)==0:
        fpr = 'DivisionByZero'    
    else:
        fpr = false_positive/(false_positive + true_negetive)
    

        
        return accuracy, precision, recall, tpr, fpr



# Plot ROC Curve from FPR and TPR values. 
def plot_roc_curve(fpr_lst, tpr_lst):

    plt.plot(fpr_lst, tpr_lst)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC-Curve")
    
    plt.savefig('ROC-Curve.png')


# Plot system states and predicted states
def plot_states_curve(original, predicted, timestamps):
    
    x = timestamps
    y1 = original
    y2 = predicted
    
    plt.figure(figsize=(50,2))
    plt.plot(x, y1, c='blue', linewidth=0.5)
    plt.plot(x, y2, c='red', linewidth=0.5)
    plt.xlabel('Time')
    plt.ylabel('States')
    
    #reduce the number of xticks.
    interval = int(len(x)/20)
    #rotate the xtick label so it takes less space.
    plt.xticks(x[::interval], rotation=90)
    plt.title("States(blue for true state and red for predicted value)",fontsize=10)
    plt.savefig('State-Curve.png')


# Using the classifier to predict system states.
def classifier_predict(data, reconstructed, threshold):
    predicted = []
    
    for v1, v2 in zip(data, reconstructed):
        pred_state = classifier(v1, v2, threshold)
        predicted.append(pred_state)
    
    
    return predicted


# Preprocessing of input sensor readings so they can be fed into the autoencoder
def sensor_mode_preprocessing(version, train_df, test_df, label_df):
    if version == '23.05':
        train_df = train_df[train_df.columns[train_df.columns!='timestamp']]
        test_df = test_df[test_df.columns[test_df.columns!='timestamp']]
        label_df = label_df[label_df.columns[label_df.columns!='timestamp']]
        

    elif version == 'end23.05':
        train_df = train_df[train_df.columns[train_df.columns!='Timestamp']]
        test_df = test_df[test_df.columns[test_df.columns!='Timestamp']]
        label_df = label_df['label']
        

        
    elif version == '22.04':
        train_df = train_df[train_df.columns[train_df.columns!='timestamp']]
        test_df = test_df[test_df.columns[test_df.columns!='timestamp']]
        label_df = test_df[test_df.columns[test_df.columns=='Attack']]
        
    train_arr = train_df.to_numpy()
    test_arr = test_df.to_numpy()
    label_arr = label_df.to_numpy()
        
    return train_arr, test_arr, label_arr    
        

#Preprocessing of input network packets so they can be fed into autoencoder.
def packets_preprocessing(packets):
    pkt_lst=[]
    for packet in packets:
        pkt = list(raw(packet))
        pkt_lst.append(pkt)
        
    pkt_arr = np.asarray(pkt_lst)
    
    return pkt_arr

#Parsing of terminal arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-v", "--version", help="Specify the database version. Support '22.04','23.05','end23.05'.")
argParser.add_argument("-m", "--mode", help="Choose running mode. Can be 'sensor' or 'packet'.")
argParser.add_argument("-f", "--file", nargs=2, help="The csv or pcap files to be read. Input train and test data in order.")
argParser.add_argument("-l", "--label", help="The label csv files for corresponding test data, if there is one.")

args = argParser.parse_args()


train_file, test_file = args.file[0], args.file[1]
# Diffrent mode will read input data differently
if args.mode == 'packet':
   
    train_pkts = rdpcap(train_file)
    test_pkts = rdpcap(test_file)
    
    train_arr = packets_preprocessing(train_pkts)
    test_arr = packets_preprocessing(test_pkts)
    
    label_df = pd.read_csv(args.label)
    new_label_df = label_df[label_df.columns[label_df.columns!='timestamp']]
    label_arr = new_label_df.to_numpy()

elif args.mode == 'sensor':    

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    if (args.version == '23.05') or (args.version == 'end23.05'):
    	label_df = pd.read_csv(args.label)
    else:
    	label_df = None

    train_arr, test_arr, label_arr = sensor_mode_preprocessing(args.version, train_df, test_df, label_df)

# The dimension of input data
n = len(train_arr[0])


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

#Build model 2
input_data_2 = keras.Input(shape=(n,))
encoded_2 = keras.layers.Dense(64, activation='relu')(input_data_2)
encoded_2 = keras.layers.Dense(32, activation='softmax')(encoded_2)
encoded_2 = keras.layers.Dense(16, activation='sigmoid')(encoded_2)

decoded_2 = keras.layers.Dense(32, activation='sigmoid')(encoded_2)
decoded_2 = keras.layers.Dense(64, activation='softmax')(decoded_2)
decoded_2 = keras.layers.Dense(n, activation='relu')(decoded_2)

autoencoder_2 = keras.Model(input_data_2,decoded_2)
autoencoder_2.compile(optimizer='adam', loss='mean_squared_error')
encoder_2 = keras.Model(input_data_2, encoded_2)

#Build model 3
input_data_3 = keras.Input(shape=(n,))
encoded_3 = keras.layers.Dense(64, activation='relu')(input_data_3)
encoded_3 = keras.layers.Dropout(0.2)(encoded_3)
encoded_3 = keras.layers.Dense(32, activation='relu')(encoded_3)
encoded_3 = keras.layers.Dropout(0.2)(encoded_3)
encoded_3 = keras.layers.Dense(16, activation='softmax')(encoded_3)
encoded_3 = keras.layers.Dropout(0.2)(encoded_3)

decoded_3 = keras.layers.Dense(32, activation='relu')(encoded_3)
decoded_3 = keras.layers.Dropout(0.2)(decoded_3)
decoded_3 = keras.layers.Dense(64, activation='relu')(decoded_3)
decoded_3 = keras.layers.Dropout(0.2)(decoded_3)
decoded_3 = keras.layers.Dense(n, activation='sigmoid')(decoded_3)

autoencoder_3 = keras.Model(input_data_3,decoded_3)
autoencoder_3.compile(optimizer='adam', loss='mean_squared_error')
encoder_3 = keras.Model(input_data_3, encoded_3)

#Build model 4
input_data_4 = keras.Input(shape=(n,))
encoded_4 = keras.layers.Dense(32, activation='relu')(input_data_4)
encoded_4 = keras.layers.Dense(16, activation='relu')(encoded_4)
encoded_4 = keras.layers.Dense(8, activation='sigmoid')(encoded_4)

decoded_4 = keras.layers.Dense(16, activation='softmax')(encoded_4)
decoded_4 = keras.layers.Dense(32, activation='relu')(decoded_4)
decoded_4 = keras.layers.Dense(n, activation='relu')(decoded_4)

autoencoder_4 = keras.Model(input_data_4,decoded_4)
autoencoder_4.compile(optimizer='adam', loss='mean_squared_error')
encoder_4 = keras.Model(input_data_4, encoded_4)

#Build model 5
input_data_5 = keras.Input(shape=(n,))
encoded_5 = keras.layers.Dense(64)(input_data_5)
encoded_5 = keras.layers.Dropout(0.2)(encoded_5)
encoded_5 = keras.layers.Activation('sigmoid')(encoded_5)
encoded_5 = keras.layers.Dense(32)(encoded_5)
encoded_5 = keras.layers.Dropout(0.2)(encoded_5)
encoded_5 = keras.layers.Activation('relu')(encoded_5)
encoded_5 = keras.layers.Dense(16)(encoded_5)
encoded_5 = keras.layers.Dropout(0.2)(encoded_5)

decoded_5 = keras.layers.Dense(32)(encoded_5)
decoded_5 = keras.layers.Dropout(0.2)(decoded_5)
decoded_5 = keras.layers.Dense(64)(decoded_5)
decoded_5 = keras.layers.Dropout(0.2)(decoded_5)
encoded_5 = keras.layers.Activation('softmax')(decoded_5)
decoded_5 = keras.layers.Dense(n)(decoded_5)

autoencoder_5 = keras.Model(input_data_5,decoded_5)
autoencoder_5.compile(optimizer='adam', loss='mean_squared_error')
encoder_5 = keras.Model(input_data_5, encoded_5)

#Build model 6
input_data_6 = keras.Input(shape=(n,))
encoded_6 = keras.layers.Dense(32, activation='relu')(input_data_6)
encoded_6 = keras.layers.GaussianNoise(1)(encoded_6)
encoded_6 = keras.layers.Dense(16, activation='relu')(encoded_6)
encoded_6 = keras.layers.GaussianNoise(1)(encoded_6)
encoded_6 = keras.layers.Dense(8, activation='softmax')(encoded_6)
encoded_6 = keras.layers.GaussianNoise(1)(encoded_6)

decoded_6 = keras.layers.Dense(16, activation='relu')(encoded_6)
decoded_6 = keras.layers.GaussianNoise(1)(decoded_6)
decoded_6 = keras.layers.Dense(32, activation='relu')(decoded_6)
decoded_6 = keras.layers.GaussianNoise(1)(decoded_6)
decoded_6 = keras.layers.Dense(n, activation='sigmoid')(decoded_6)

autoencoder_6 = keras.Model(input_data_6,decoded_6)
autoencoder_6.compile(optimizer='adam', loss='mean_squared_error')
encoder_6 = keras.Model(input_data_6, encoded_6)

#Ask user to choose one of the models to use.
chosen_model = input("Choose a model to be used: (from 1 to 6)\n ")

#For the chosen model, calculate compression factor and print some information
if chosen_model == '1':
    comp_factor = 16/n
    enc_layer_num = 3
    dec_layer_num = 3
    print("Some information of this model:\n")
    print("Compression factor:"+str(comp_factor)+"\n")
    print("Number of layers in encoder:"+str(enc_layer_num)+"\n")    
    print("Number of layers in decoder:"+str(dec_layer_num)+"\n")
    
    autoencoder = autoencoder_1
    encoder = encoder_1
    
elif chosen_model == '2':
    comp_factor = 16/n
    enc_layer_num = 3
    dec_layer_num = 3
    print("Some information of this model:\n")
    print("Compression factor:"+str(comp_factor)+"\n")
    print("Number of layers in encoder:"+str(enc_layer_num)+"\n")    
    print("Number of layers in decoder:"+str(dec_layer_num)+"\n")
    
    autoencoder = autoencoder_2
    encoder = encoder_2
    
elif chosen_model == '3':
    comp_factor = 16/n
    enc_layer_num = 6
    dec_layer_num = 5
    print("Some information of this model:\n")
    print("Compression factor:"+str(comp_factor)+"\n")
    print("Number of layers in encoder:"+str(enc_layer_num)+"\n")    
    print("Number of layers in decoder:"+str(dec_layer_num)+"\n")
    
    autoencoder = autoencoder_3
    encoder = encoder_3    

elif chosen_model == '4':
    comp_factor = 8/n
    enc_layer_num = 3
    dec_layer_num = 3
    print("Some information of this model:\n")
    print("Compression factor:"+str(comp_factor)+"\n")
    print("Number of layers in encoder:"+str(enc_layer_num)+"\n")    
    print("Number of layers in decoder:"+str(dec_layer_num)+"\n")
    
    autoencoder = autoencoder_4
    encoder = encoder_4
    
elif chosen_model == '5':
    comp_factor = 16/n
    enc_layer_num = 8
    dec_layer_num = 6
    print("Some information of this model:\n")
    print("Compression factor:"+str(comp_factor)+"\n")
    print("Number of layers in encoder:"+str(enc_layer_num)+"\n")    
    print("Number of layers in decoder:"+str(dec_layer_num)+"\n")
    
    autoencoder = autoencoder_5
    encoder = encoder_5

elif chosen_model == '6':
    comp_factor = 8/n
    enc_layer_num = 6
    dec_layer_num = 5
    print("Some information of this model:\n")
    print("Compression factor:"+str(comp_factor)+"\n")
    print("Number of layers in encoder:"+str(enc_layer_num)+"\n")    
    print("Number of layers in decoder:"+str(dec_layer_num)+"\n")
    
    autoencoder = autoencoder_6
    encoder = encoder_6


#Normalization of training data
min_max_scaler = preprocessing.MinMaxScaler()
scaled_train_arr = min_max_scaler.fit_transform(train_arr)
scaled_test_arr = min_max_scaler.transform(test_arr)

#Training the model

autoencoder.fit(scaled_train_arr, scaled_train_arr, epochs=10, validation_split=0.2)

#Generating features 

print("Generating features for train and test data\n")
train_features = encoder.predict(scaled_train_arr)
test_features = encoder.predict(scaled_test_arr)

#Save results to csv files

train_feature_df = pd.DataFrame(train_features)
test_feature_df = pd.DataFrame(test_features)
train_feature_df.to_csv("model-"+chosen_model+"-train-features.csv")
test_feature_df.to_csv("model-"+chosen_model+"-test-features.csv")


print("features saved to files\n\n\n")

#The codes below inplements classifier functions. Including calculation of accuracy, precision, recall, true positive rate, false positive rate and plot of ROC Curve and system states. 
print("Now implements the classifier function\n")

#Take threshold from user input.
threshold = input("give a threshold for the classifier:\n")
threshold = float(threshold)

#Reconstruction of test data
recons_test_arr = autoencoder.predict(scaled_test_arr)
#Classifier implementation
pred_arr = classifier_predict(scaled_test_arr, recons_test_arr, threshold)
#Calculation of accuracy, precision and recall
accuracy, precision, recall,_,_ = metric_calculation(label_arr, pred_arr) 
#print out the result.
print("Accuracy of the result:"+str(accuracy)+"\n")
print("Precision of the result:"+str(precision)+"\n")
print("Recall of the result:"+str(recall)+"\n")
print("\n")


#Now take the thresholds to calculate some pairs of tpr and fpr in order to plot ROC Curve
print("For the generation of ROC Curve, please enter following parameters:\n")
lower = input("lower bound of threshold:\n")
upper = input("upper bound of threshold:\n")
step = input("step size to be taken:\n")

lower = float(lower)
upper = float(upper)
step = float(step)

#The list of thresholds is generated using numpy.arange(). Parameters are chosen by user before.
tprs,fprs = generate_roc_prameters(scaled_test_arr, recons_test_arr, label_arr, np.arange(lower,upper,step))

#The flag to decide if all fprs and tprs have valid value(Not divison by zero)
flag = 1
for tpr, fpr in zip(tprs, fprs):
    if(tpr == 'DivisionByZero') or (fpr == 'DivisionByZero'):
        print("not appropriate thresholds\n")
        flag = 0
        break
#Plot the ROC Curve only if values are valid        
if (flag == 1):
    print("Creating graph of ROC Curve...\n")
    plot_roc_curve(fprs,tprs)


print("Creating gragh of states of test data and predicted result")
#Take timestamps in order to plot system states and predicted states.
if label_df == None:
    timestamp_lst = np.reshape(test_df[test_df.columns[test_df.columns=='timestamp']].values,-1).tolist()
else:
    timestamp_lst = np.reshape(label_df[label_df.columns[label_df.columns=='timestamp']].values,-1).tolist()

plot_states_curve(label_arr,pred_arr,timestamp_lst)

