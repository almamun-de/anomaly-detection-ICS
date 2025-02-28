
Used libararies and tools for this task :
    pandas
    numpy
    scikit-learn
    argparse
    matplotlib


##### Description  #####
This code reads csv files of train and test datasets into pandas dataframe and perform some preprocessing like normalization. Then for each sensor, calculates the K-S statistic with and without considering system states. In the meantime the number of sensors pass K-S test with and without considering system states and percentage of common system states in train and test sets are also calculated and plotted. 

For each pair of train and test set, the results will be stored into one csv file containing K-S statistic with and without considering system states for all sensors, and two png files containing the plots for subtask e).

Generated results are shared with Tutor Asya Mitseva by BTU cloud.

All the intermediate results (dataframe, list, variables) could be accessed by running the Demo_Task2.ipynb file in jupyter notebook.




############################################################
######   Usage  #####
For task3 the codes in 'generate_packets.py' is used.
Put it under the same folder with the datasets before running.
Run it on terminal with 'python generate_packets.py -h' for detailed usage.


##### References #####
Used libararies and tools for this task :
    pandas
    numpy
    scapy
    struct
    binascii
    argparse
    cip-enipTCP project for scapy from Github: https://github.com/scy-phy/scapy-cip-enip/tree/master 
    Source of legitmate EtherNet/IP packets: https://github.com/ITI/ICS-Security-Tools/tree/master/pcaps/EthernetIP


##### Description #####
This code replace the payload from legitmate EtherNet/IP packet with the sensor readings from HAI datasets and repack them into new packets.

For each train or test dataset, the packets containing physical readings for each sensor or acturator are stored into one pcap file.(If there are n sensors and acturators in one train set, then n pcap files would be generated for this train set.)

Generated results are shared with Tutor Asya Mitseva by BTU cloud.

All the intermediate results (dataframe, list, variables) could be accessed by running the Demo_Task3.ipynb file in jupyter notebook.



