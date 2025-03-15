
# Argument parser setup
parser = argparse.ArgumentParser(description="Process sensor data from a given dataset version.")
parser.add_argument('-f', '--folder', type=str, required=True, help="Path to the dataset version folder. Example: hai-22.04")
args = parser.parse_args()

# Extracting the dataset version from the folder path
dataset_folder = args.folder
dataset_version = os.path.basename(dataset_folder)

# Define the dataset structure
datasets = {
    "hai-22.04": ["test1.csv", "test2.csv", "test3.csv", "test4.csv",
                  "train1.csv", "train2.csv", "train3.csv", "train4.csv", "train5.csv", "train6.csv"],
    "hai-23.05": ["hai-test1-label.csv", "hai-test2-label.csv",
                  "hai-train1.csv", "hai-train2.csv", "hai-train3.csv", "hai-train4.csv"],
    "haiend-23.05": ["end-test1-label.csv", "end-test2-label.csv",
                     "end-train1.csv", "end-train2.csv", "end-train3.csv", "end-train4.csv"]
}

if dataset_version not in datasets:
    raise ValueError(f"Dataset version '{dataset_version}' is not recognized.")

def refined_plot_sensor_data(df, attacked_timestamps, columns, dataset_name, file_name):
    """Plot sensor/actuator data in a refined style and mark sections under attack."""
    for column in columns:
        if column not in ["timestamp", "Timestamp", "label", "Attack"]:
            plt.figure(figsize=(18, 6))
            timestamp_col = 'Timestamp' if 'Timestamp' in df.columns else 'timestamp'
            plt.plot(df[timestamp_col], df[column], label='Measurement', color='blue', linewidth=0.7)

            # Mark the attack periods with red vertical lines
            for ts in attacked_timestamps:
                plt.axvline(x=ts, color='red', linewidth=0.5)

            # Additional styling
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.title(f"{dataset_name} - {file_name} - {column}")
            plt.xlabel('Timestamp')
            plt.ylabel('Value')

            # Reduce the number of x-ticks and rotate them
            ticks = plt.xticks()[0]  # The positions of the current x-ticks.
            step = len(ticks) // 30  # Reduce the number of ticks to approximately 30
            plt.xticks(ticks[::step], rotation=90)  # Reducing the number of x-ticks.

            plt.tight_layout()  # Adjust Layout automatically.

            # Save the plot
            plt.savefig(f"{dataset_name}_{file_name}_{column}.png")
            plt.close()  # close the current figure window.

def export_to_pdf(dataset_name, dataset_folder):
    """Export plots related to a dataset to a single PDF file."""
    pdf_file = f"{dataset_folder}/{dataset_name}_plots.pdf"
    with PdfPages(pdf_file) as pdf:
        for file in glob.glob(f"{dataset_folder}/{dataset_name}_*.png"):
            plt.figure(figsize=(18, 6))
            img = plt.imread(file)
            plt.imshow(img)
            plt.axis('off')
            pdf.savefig()
            plt.close()
            # os.remove(file)  # Optionally remove the PNG file after adding to PDF


# Process each dataset
for file in datasets[dataset_version]:
    filepath = os.path.join(dataset_folder, file)
    df = pd.read_csv(filepath)

    if dataset_version == "hai-22.04":
        attacked_timestamps = df[df['Attack'] == 1]['timestamp' if 'timestamp' in df.columns else 'Timestamp'].tolist()
    elif "label" in file:
        attacked_timestamps = df[df['label'] == 1]['timestamp' if 'timestamp' in df.columns else 'Timestamp'].tolist()
    else:
        attacked_timestamps = []

    refined_plot_sensor_data(df, attacked_timestamps, df.columns, dataset_version, file)

export_to_pdf(dataset_version, dataset_folder)
