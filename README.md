# Quantum-Classical_Hybrid_Q-Kernel_MultiClass-Classifier-Emotion_detection
Quantum-Classical_Hybrid_Neural Network Multi Class-Classifier : Emotion detection model, based on Pennylane. 


This repository contains the implementation of a Quantum-Classical Hybrid Machine Learning Model that combines BERT embeddings with a Quantum Kernel-based Support Vector Machine (SVM) to perform emotion classification using the GoEmotions dataset.

This particular version uses 250 samples for rapid prototyping and demonstration. The repository also includes experiments with 1000-sample, 5000-sample, and 53,000-sample datasets, along with failed iterations (like Decision Tree models) and a dedicated log file explaining the rationale behind each iteration and why some approaches did not succeed.

# Project Structure
text
Copy
Edit
Quantum-Classical_Hybrid_ML/
│
├── 250_sample_version/             # Current working version (demo) - 250 samples
├── 1000_sample_version/            # Scaled version with 1000 samples
├── 5000_sample_version/            # Scaled version with 5000 samples
├── 53000_sample_version/           # Full dataset version (GoEmotions - 53k samples)
│
├── failed_iterations/              # Includes failed experiments (Decision Trees, etc.)
│   ├── decision_tree_attempt.py
│   └── ...
│
├── iteration_logs/                 # Logs and reasoning for each iteration and approach
│   └── iteration_reasoning.txt
│
├── README.md                       # Project documentation
└── requirements.txt                # Python package dependencies
# Description
The goal of this project is to explore the potential of quantum-enhanced machine learning using quantum kernels on NLP-based emotion classification tasks. By leveraging BERT's sentence embeddings and quantum feature maps, we aim to test whether quantum kernels can provide an advantage over classical models in low-data regimes.

# Highlights:
✅ Quantum-Classical Hybrid Workflow

✅ BERT Embedding Extraction

✅ PCA Dimensionality Reduction

✅ Quantum Kernel Calculation using Pennylane

✅ SVM Classifier with Precomputed Quantum Kernel

✅ Supports Scalable Versions: 250, 1000, 5000, and 53,000 samples

✅ Detailed Logs of Failed Attempts and Their Reasons

# Available Versions
Version	Purpose	Location
250-sample version	Fast prototype & demonstration	250_sample_version/
1000-sample version	Intermediate scale testing	1000_sample_version/
5000-sample version	High computational version	5000_sample_version/
53,000-sample version	Full dataset experiment	53000_sample_version/
Failed Iterations	Decision Trees, others	failed_iterations/
Iteration Logs	Detailed reasoning per attempt	iteration_logs/

# Dataset
Dataset: GoEmotions Dataset

Classes: 28 emotion labels

# Technologies Used
Python 🐍

Pennylane

Scikit-Learn

Transformers (BERT)

TensorFlow Datasets

JAX

Matplotlib & Seaborn

# Setup Instructions
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/JhaAyushCanCode/Quantum-Classical_Hybrid_ML.git
cd Quantum-Classical_Hybrid_ML
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
# Note: For Colab users, a custom environment setup using specific JAX and Pennylane versions is required (see the notebook for pip commands).

# Key Components
Data Preprocessing: Loading GoEmotions dataset and extracting BERT embeddings

Dimensionality Reduction: PCA to reduce BERT output to 16 dimensions

Quantum Feature Mapping: Encoding classical data into quantum states

Quantum Kernel Construction: Computing kernel matrices using Pennylane

Classification: SVM using precomputed quantum kernels

Visualization: Accuracy plot and Confusion Matrix

# Experiments and Logs
The repository provides:

Detailed logs explaining every iteration, parameter choices, model selection, and failures.

Failed experiments folder containing non-performing classical models like Decision Trees.

Progression from small to full datasets to observe scalability and performance trends.

# Example Output
Test Accuracy: Visualized using bar plots.

Confusion Matrix: Displayed using Seaborn heatmaps.

Classification Report: Precision, Recall, and F1-Score per class.

# Future Directions
Implement noise-aware quantum simulators or real quantum hardware.

Integrate multi-class quantum classifiers.

Compare quantum kernel performance against other advanced classical kernels.

Optimize quantum circuit depth to reduce computational load.

# Contribution
Feel free to fork, explore, and contribute by creating pull requests. Check the iteration logs to understand the evolution of the project.

# License
This project is licensed under the MIT License.
