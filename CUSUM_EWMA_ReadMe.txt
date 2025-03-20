######## BEFORE RUNNING THE PROGRAM ##########

1.Make sure the csv file to be read is under same directory of the script. 

2.The following library is requred:
	pandas
	numpy
	functools
	argparser



####### USAGE #######

To get  help information run

$ python 2-2.py -h  

Normal usage run

$ python 2-2.py -f [training data] [testing data] -m [method] -t [threshold]

Note method must be either 'CUSUM' or 'EWMA', threshold must be float point number.


####### RESULT #######

two csv files will be generated after each execution.

one containig parameters trained from training data.

another containing alarms results from testing data.
