#!/usr/bin/env python
# coding: utf-8


def ks_statistic(obs_one, obs_two):
    cdf_one = np.sort(obs_one)
    cdf_two = np.sort(obs_two)

    i = 0
    j = 0
    d = 0.0
    fn1 = 0.0
    fn2 = 0.0
    l1 = float(len(cdf_one))
    l2 = float(len(cdf_two))

    while (i < len(cdf_one) and j < len(cdf_two)):
        d1 = cdf_one[i]
        d2 = cdf_two[j]
        if d1 <= d2:
            i = i + 1
            fn1 = i/l1
        if d2 <= d1:
            j = j + 1
            fn2 = j/l2
        dist = abs(fn2 - fn1)
        if dist > d:
            d = dist

    return d



def normalize(train, test, scaler):

    
    normalized_train = pd.DataFrame(
        scaler.fit_transform(train),
        columns = train.columns
    )
    
    normalized_test = pd.DataFrame(
        scaler.transform(test),
        columns = test.columns
    )
    
    
    
    return normalized_train, normalized_test


def ks_result(ks_train_df, ks_test_df):

    ks_list = []
    for sensor in ks_train_df.columns:

        n = ks_statistic(ks_train_df[sensor],ks_test_df[sensor])
        ks_list.append([sensor,n])

    ks_df = pd.DataFrame(ks_list, columns = ['sensor','ks_statistic'])
    
    return ks_df




def get_common_states(acturater_list, normalized_train_df, normalized_test_df):

    train_sys_states = list(normalized_train_df.value_counts(acturator_list).index)
    test_sys_states = list(normalized_test_df.value_counts(acturator_list).index)

    common_states = list(set(train_sys_states).intersection(test_sys_states))
    
    return common_states



def extract_reading_by_state(acturator_list, common_state, df):
    
    new_df = df
    i=0
    for acturator in acturator_list:
        new_df =  new_df[new_df[acturator]==common_state[i]]
        i = i+1
    
    return new_df



def ks_test(df1, df2, acturator_list):
    
    sensor_df1 =df1[df1.columns[~df1.columns.isin(acturator_list)]]
    sensor_df2 =df2[df2.columns[~df2.columns.isin(acturator_list)]]
    
    ks_df = ks_result(sensor_df1, sensor_df2)
    ks_sum_df = ks_result(sensor_df1, sensor_df2)
    ks_sum_df['ks_statistic'] -= ks_sum_df['ks_statistic']
    
    common_states = get_common_states(acturator_list, df1, df2)
    for state in common_states:
        new_df1 = extract_reading_by_state(acturator_list, state, df1)
        new_df2 = extract_reading_by_state(acturator_list, state, df2)
        
        new_sensor_df1 = new_df1[new_df1.columns[~new_df1.columns.isin(acturator_list)]]
        new_sensor_df2 = new_df2[new_df2.columns[~new_df2.columns.isin(acturator_list)]]
        
        ks_sum_df['ks_statistic'] += ks_result(new_sensor_df1, new_sensor_df2)['ks_statistic']
        
    ks_sum_df = ks_sum_df.rename(columns={'ks_statistic':'ks_by_states'})
    ks_sum_df['ks_by_states'] /= len(common_states)
    
    ks_df = ks_df.merge(ks_sum_df,how="left", on='sensor')
    
    return ks_df
        

def count_sensor(ks_df):
    count1 = len(ks_df[ks_df['ks_statistic']<=0.17])
    count2 = len(ks_df[ks_df['ks_by_states']<=0.17])
    
    return count1, count2



def percent_common_states(acturator_list, df1, df2):
    common_states = get_common_states(acturator_list, df1, df2)
    states1 = list(df1.value_counts(acturator_list).index)
    states2 = list(df2.value_counts(acturator_list).index)
    per_1 = len(common_states)/len(states1)
    per_2 = len(common_states)/len(states2)
    per_3 = len(common_states)/(len(states1)+len(states2)-len(common_states))
    
    return per_1, per_2, per_3


import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt


argParser = argparse.ArgumentParser()
argParser.add_argument("-v", "--version", help="Specify the database version. Support '22.04','23.05','end23.05'.")
argParser.add_argument("-s", "--scaler", help="scaling method. Use 'standard' or 'min-max'. Default 'standard'.")
argParser.add_argument("-f", "--file", nargs=2, help="The csv files to be read. Input train and test data in order.")
argParser.add_argument("-l", "--label", help="The label csv files for corresponding test data, if there is one.")

args = argParser.parse_args()


train, test = args.file[0],args.file[1]

train_df = pd.read_csv(train)
test_df = pd.read_csv(test)


scaler = preprocessing.StandardScaler()

if(args.scaler == "min-max"):
    
    scaler = preprocessing.MinMaxScaler()



if args.version == '23.05':

    label_df = pd.read_csv(args.label)

    acturator_list = ['P1_PP01AD','P1_PP01AR','P1_PP01BD','P1_PP01BR','P1_PP02D','P1_PP02R','P1_SOL01D','P1_SOL03D','P1_STSP','P2_ATSW_Lamp','P2_AutoGO','P2_Emerg','P2_MASW','P2_MASW_Lamp','P2_ManualGO','P2_OnOff','P2_TripEx']

#Remove timestamp column since it is not to be processed.
    train_df = train_df[train_df.columns[train_df.columns!='timestamp']]
    test_df = test_df[test_df.columns[test_df.columns!='timestamp']]
    label_df = label_df[label_df.columns[label_df.columns!='timestamp']]
    
#Columns that are sensor readings.
    train_sensor_df = train_df[train_df.columns[~train_df.columns.isin(acturator_list)]]
    test_sensor_df = test_df[test_df.columns[~test_df.columns.isin(acturator_list)]]

#Columns that are acturator states.
    train_acturator_df = train_df[train_df.columns[train_df.columns.isin(acturator_list)]]
    test_acturator_df = test_df[test_df.columns[test_df.columns.isin(acturator_list)]]

#Normalize of sensor readings.        
    ks_train_df, ks_test_df = normalize(train_sensor_df, test_sensor_df, scaler)

#Merge normalized sensor readings with acturators.
    ks_train_df = ks_train_df.merge(train_acturator_df,left_index=True, right_index=True)
    ks_test_df = ks_test_df.merge(test_acturator_df,left_index=True, right_index=True)

#Remove rows that are under attack.
    ks_test_df = ks_test_df.merge(label_df,left_index=True, right_index=True)
    ks_test_df = ks_test_df[ks_test_df['label'] == 0]
    ks_test_df = ks_test_df[ks_test_df.columns[ks_test_df.columns!='label']]

#Perform calculation of ks-statistic with and without considering states.
    ks_df = ks_test(ks_train_df, ks_test_df, acturator_list)
    filename = 'hai23.05-'+args.file[0]+'-'+args.file[1]
    filename1 =filename+'-ks.csv'
    ks_df.to_csv(filename1)


    
    
elif args.version == 'end23.05':

    label_df = pd.read_csv(args.label)

    acturator_list = ['DM-HT01-D','DM-LCV01-MIS','DM-LSH-03','DM-LSH-04','DM-LSH01','DM-LSH02','DM-LSL-04','DM-LSL01','DM-LSL02','DM-PCV01-DEV','DM-PP01-R','DM-PP01A-D','DM-PP01B-D','DM-PP01A-R','DM-PP01B-R','DM-PP02-D','DM-PP02-R','DM-PP04-D','DM-SOL01-D','DM-SOL02-D','DM-SOL03-D','DM-SOL04-D','DM-SS01-RM','DM-ST-SP','DM-SW01-ST','DM-SW02-SP','DM-SW03-EM','DQ03-LCV01-D','DQ04-LCV01-DEV']


    train_df = train_df[train_df.columns[train_df.columns!='Timestamp']]
    test_df = test_df[test_df.columns[test_df.columns!='Timestamp']]
    label_df = label_df[label_df.columns[label_df.columns!='Timestamp']]

    train_sensor_df = train_df[train_df.columns[~train_df.columns.isin(acturator_list)]]
    test_sensor_df = test_df[test_df.columns[~test_df.columns.isin(acturator_list)]]

    train_acturator_df = train_df[train_df.columns[train_df.columns.isin(acturator_list)]]
    test_acturator_df = test_df[test_df.columns[test_df.columns.isin(acturator_list)]]

        
    ks_train_df, ks_test_df = normalize(train_sensor_df, test_sensor_df, scaler)

    ks_train_df = ks_train_df.merge(train_acturator_df,left_index=True, right_index=True)
    ks_test_df = ks_test_df.merge(test_acturator_df,left_index=True, right_index=True)


    ks_test_df = ks_test_df.merge(label_df,left_index=True, right_index=True)
    ks_test_df = ks_test_df[ks_test_df['label'] == 0]
    ks_test_df = ks_test_df[ks_test_df.columns[ks_test_df.columns!='label']]


    ks_df = ks_test(ks_train_df, ks_test_df, acturator_list)
    filename = 'end23.05-'+args.file[0]+'-'+args.file[1]
    filename1 =filename +'-ks.csv'
    ks_df.to_csv(filename1)

    
elif args.version == '22.04':

    
    acturator_list = ['P1_PP01AD','P1_PP01AR','P1_PP01BD','P1_PP01BR','P1_PP02D','P1_PP02R','P1_SOL01D','P1_SOL03D','P1_STSP','P2_ATSW_Lamp','P2_AutoGO','P2_Emerg','P2_MASW','P2_MASW_Lamp','P2_ManualGO','P2_OnOff','P2_TripEx']


    train_df = train_df[train_df.columns[train_df.columns!='timestamp']]
    test_df = test_df[test_df.columns[test_df.columns!='timestamp']]

    train_sensor_df = train_df[train_df.columns[~train_df.columns.isin(acturator_list)]]
    test_sensor_df = test_df[test_df.columns[~test_df.columns.isin(acturator_list)]]

    train_acturator_df = train_df[train_df.columns[train_df.columns.isin(acturator_list)]]
    test_acturator_df = test_df[test_df.columns[test_df.columns.isin(acturator_list)]]

        
    ks_train_df, ks_test_df = normalize(train_sensor_df, test_sensor_df, scaler)

    ks_train_df = ks_train_df.merge(train_acturator_df,left_index=True, right_index=True)
    ks_test_df = ks_test_df.merge(test_acturator_df,left_index=True, right_index=True)


    ks_test_df = ks_test_df[ks_test_df['Attack'] == 0]
    ks_test_df = ks_test_df[ks_test_df.columns[ks_test_df.columns!='Attack']]

    ks_train_df = ks_train_df[ks_train_df['Attack'] == 0]
    ks_train_df = ks_train_df[ks_train_df.columns[ks_train_df.columns!='Attack']]

    ks_df = ks_test(ks_train_df, ks_test_df, acturator_list)
    filename = 'hai22.04-'+args.file[0]+'-'+args.file[1]
    filename1 = filename+'-ks.csv'
    ks_df.to_csv(filename1)
    
    
else:
    
    print("Please specify version and run again.")
    
#Plot some results.        
    
c1, c2 = count_sensor(ks_df)
p1, p2, p3 = percent_common_states(acturator_list, ks_train_df, ks_test_df)
    
    
filename2 = filename+'-sensors.png'
x1 = ['no_states','by_states']
v1 = [c1,c2]
plt.bar(x1,v1)
plt.savefig(filename2)
plt.close()
    
filename3 = filename+'-percent.png'
x2 = ['common_by_train','common_by_test','common_by_all']
v2 = [p1,p2,p3]
plt.bar(x2,v2)
plt.savefig(filename3)
plt.close()
