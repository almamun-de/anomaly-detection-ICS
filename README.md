# Anomaly Detection in Industrial Control Systems (ICS)

📌 Project Overview

This project focuses on detecting anomalies in Industrial Control Systems (ICS) using machine learning and deep learning techniques. The research leverages the HAI Security Dataset and evaluates various anomaly detection models, including statistical methods, traditional machine learning classifiers, and deep learning models (CNN, ResNet, Autoencoder).

🎯 Motivation

Industrial Control Systems (ICS) are critical for infrastructure sectors like power generation, water treatment, and manufacturing. However, cyber threats targeting ICS have been increasing. Traditional IT security measures are not sufficient due to the deterministic nature of Operational Technology (OT). This study aims to improve Intrusion Detection Systems (IDS) by developing more effective anomaly detection methods using feature extraction and machine learning models.

📂 Project Structure
anomaly-detection-ICS/
├── data/                 # Dataset (or links to external sources)
├── src/                  # Source code files
│   ├── preprocess.py     # Data preprocessing
│   ├── feature_extraction.py  # Feature extraction methods (PCA, Autoencoder, N-Grams)
│   ├── train_models.py   # Training scripts (Statistical, ML, DL models)
│   ├── evaluate.py       # Model evaluation
│   ├── toolbox.py        # Toolbox for running experiments
│   ├── visualization.py  # Data visualization scripts
├── notebooks/            # Jupyter Notebooks (if applicable)
├── models/               # Saved trained models
├── results/              # Evaluation results (plots, metrics, confusion matrices)
├── README.md             # Project documentation (this file)
├── requirements.txt      # Dependencies list
├── .gitignore            # Files to exclude from Git
├── LICENSE               # License (MIT, Apache, or other)
└── report/               # Research paper

📊 Dataset

The project uses the HAI Security Dataset to train and evaluate anomaly detection models. The dataset includes sensor and actuator readings from an ICS testbed, labeled with normal and attack states.

🔗 Dataset Source: [HAI Security Dataset](https://github.com/icsdataset/hai#hai-dataset)

🛠️ Features & Techniques Used

1️⃣ Feature Extraction Methods

Principal Component Analysis (PCA): Reduces high-dimensional data while retaining important variance.

Autoencoder (Deep Learning): Compresses and reconstructs data to detect anomalies.

N-Grams: Captures sequential patterns in sensor readings.

2️⃣ Anomaly Detection Models

✅ Statistical Methods

MinMax, Gradient, Steadytime, Histogram, CUSUM, EWMA

✅ Traditional Machine Learning Classifiers

One-Class SVM, Local Outlier Factor (LOF), Elliptic Envelope (EE)

Multi-Class SVM, Random Forest (RF), k-Nearest Neighbors (KNN)

Ensemble Classifier: Combines multiple classifiers (Random, Majority, All)

✅ Deep Learning Models

Convolutional Neural Network (CNN): Captures spatial patterns in ICS data.

Residual Neural Network (ResNet): Addresses deep learning vanishing gradient issues.

Autoencoder: Learns data distribution and detects anomalies based on reconstruction errors.

🔬 Evaluation Metrics

The performance of the models is evaluated using:

Precision & Recall

Confusion Matrix

Kolmogorov-Smirnov (K-S) Statistic

🚀 Installation & Setup

🔹 Prerequisites

Ensure you have Python 3.x installed along with the required dependencies.

🔹 Install Dependencies

pip install -r requirements.txt

🔹 Run Preprocessing

python src/preprocess.py

🔹 Train a Model

python src/train_models.py --model cnn

🔹 Evaluate the Model

python src/evaluate.py --model cnn

📌 Results & Findings

Statistical detection methods showed high recall but low precision, leading to many false positives.

Autoencoder achieved high anomaly detection performance, but required tuning to reduce false positives.

CNN and ResNet models outperformed traditional classifiers in anomaly detection, making them promising for ICS security.

📝 Future Work

Fine-tune deep learning models to reduce false positives.

Test on additional ICS datasets to improve generalizability.

Implement real-time anomaly detection for ICS environments.

📜 License

This project is licensed under the MIT License – see the LICENSE file for details.

🙌 Acknowledgments

This project was conducted as part of a study project at Brandenburg University of Technology, Germany, under the supervision of Prof. Dr.-Ing. Andriy Panchenko and Asya Mitseva, M.Sc.
