######## BEFORE RUNNING THE PROGRAM ##########

1.Make sure the csv and pcap files to be read are under same directory of the script. 

2.The following libraries are requred:
	pandas
	numpy
	tensorflow
	sci-kit learn
	scapy
	matplotlib
	argparser



####### USAGE #######

To get  help information run

$ python autoencoder.py -h  

Normal usage run

$ python autoencoder.py -m MODE -f TRAIN TEST [-v VERSION] [-l LABEL]

Note MODE must be either 'packet' or 'sensor'

enample:
$ python autoencoder.py -m 'sensor' -f 'train1.csv' 'test1.csv' -v '23.05' -l 'label-test1.csv'


The other parameters needed for the training of model and the classifier function will be taken from the user during running.



####### RESULT #######

Two csv files contaning the features genereted by encoder will be generated after the execution, one for train data and another for test data.

If the input thresholds during running are appropriate, an image of ROC Curve will also be generated.

In the end an image of system states and predicted states by the classifier is generated. 
