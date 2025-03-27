
def moving_average(iterable, n):
    moving_averages = []  # List to store moving averages
    sum_iter = 0          # Sum of the current window
    iter_len = len(iterable)

    for i in range(iter_len):
        sum_iter += iterable[i]  # Add next element from iterable
        if i >= n - 1:
            # Calculate average for the current window
            moving_avg = sum_iter / n
            moving_averages.append(moving_avg)

            # Subtract element leaving the window
            sum_iter -= iterable[i - (n - 1)]

    return moving_averages



# Generate n-grams or smoothed n-grams from a Pandas series.
def create_ngrams_from_series(series, n, smooth=False):
    """ Generate n-grams or smoothed n-grams from a Pandas series. """
    if smooth:
        # Apply moving average if smoothing is required
        series = moving_average(series, n)

    # Convert series to list for easier manipulation
    series_list = list(series)

    # List to store n-grams
    ngrams = []

    # Iterate over the series list to create n-grams
    for i in range(len(series_list) - n + 1):
        # Create an n-gram as a slice of the series list
        ngram = series_list[i:i + n]

        # Append the n-gram to the list
        ngrams.append(tuple(ngram))  # Convert to tuple for consistency

    return ngrams



# Generate a DataFrame of N-grams or smoothed N-grams.
def generate_ngrams(df, n, smooth):
    ngrams_dict = {}

    for sensor in df.columns:
        ngrams_dict[sensor] = create_ngrams_from_series(df[sensor], n, smooth)
    ngrams_df = pd.DataFrame(ngrams_dict)

    return ngrams_df



def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Generate N-gram based features from a CSV file.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')
    parser.add_argument('n', type=int, help='Size of the n-gram')

    # Parse arguments
    args = parser.parse_args()
    csv_file = args.csv_file
    n = args.n

    # Load dataset
    d_f = pd.read_csv(csv_file)
    # Drop the excluded columns
    df = d_f.drop(columns=EXCLUDE_COLUMNS, errors='ignore')

    # Generate and save N-grams without smoothing
    ngrams_df = generate_ngrams(df, n, smooth=False)
    ngrams_df.to_csv(f'{csv_file}_ngrams_{n}.csv', index=False)
    print(f'N-grams without smoothing saved to {csv_file}_ngrams_{n}.csv')

    # Generate and save N-grams with smoothing
    smoothed_ngrams_df = generate_ngrams(df, n, smooth=True)
    smoothed_ngrams_df.to_csv(f'{csv_file}_smoothed_ngrams_{n}.csv', index=False)
    print(f'N-grams with smoothing saved to {csv_file}_smoothed_ngrams_{n}.csv')


if __name__ == '__main__':
    main()
    
