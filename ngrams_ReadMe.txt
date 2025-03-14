
python ngrams.py train1.csv 3


Output:
The script generates two CSV files:
One with N-grams without smoothing ([input_file_name]_ngrams_[N].csv).
One with smoothed N-grams ([input_file_name]_smoothed_ngrams_[N].csv).



---------------------------
This Python tool is designed for generating N-gram based features from parsed physical readings in a CSV file. It supports both non-smoothed and smoothed N-grams, making it suitable for preparing data for traditional machine learning classifiers.

Features
N-gram Generation: Create N-grams from the given dataset where N is a user-defined parameter.
Smoothing Option: Includes an option to apply a moving average smoothing technique to the N-grams.

Requirements
Python 3.x
Pandas library
