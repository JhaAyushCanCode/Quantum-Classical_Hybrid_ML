"""
Original file is located at
    https://colab.research.google.com/drive/13nLO6CLn2s7fkSUH6H33TeE02rqmB-B7
"""

# Quantum-Classical Hybrid Model using BERT and Quantum Kernel SVM for GoEmotions Dataset

# Step 1: Environment 
!pip uninstall -y jax jaxlib
!pip install jax==0.4.28 jaxlib==0.4.28 --quiet
!pip install pennylane seaborn tensorflow-datasets scikit-learn==1.6.1 transformers --upgrade --quiet

# Step 2: Importing Libraries
import pennylane as qml
from pennylane import numpy as np
import numpy as onp
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from transformers import BertTokenizer, BertModel
import tensorflow_datasets as tfds
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Step 3: Load Dataset
print("Loading GoEmotions dataset...")
dataset, info = tfds.load('goemotions', with_info=True)
train_dataset = dataset['train']

# Emotion labels in order
emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'neutral',
    'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise'
]

texts, labels = [], []
for example in tfds.as_numpy(train_dataset):
    texts.append(example['comment_text'].decode('utf-8'))
    for idx, label in enumerate(emotion_labels):
        if example[label]:
            labels.append(idx)
            break
    else:
        labels.append(20)

n_classes = 28

# Step 4: BERT Embedding Extraction
print("Extracting BERT embeddings...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to compute BERT [CLS] embeddings
def bert_embed(sentences):
    with torch.no_grad():
        inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=128)
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()

batch_size = 64
bert_embeddings = []
for i in range(0, len(texts), batch_size):
    bert_embeddings.append(bert_embed(texts[i:i+batch_size]))
X = onp.vstack(bert_embeddings)
y = onp.array(labels)

# Step 5: PCA (Dimensionality Reduction)
print("Reducing BERT embedding dimensions using PCA...")
pca = PCA(n_components=16)
X_reduced = pca.fit_transform(X)

# Limit to 5000 samples -> computational load reduction
X = X[:5000]
y = y[:5000]

# Step 5: Dimensionality Reduction
print("Reducing BERT embedding dimensions using PCA...")
pca = PCA(n_components=16)
X_reduced = pca.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Step 6: Quantum Kernel Setup
n_qubits = 16
dev = qml.device('default.qubit', wires=n_qubits)

def feature_map(x):
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    qml.CNOT(wires=[n_qubits - 1, 0])

@qml.qnode(dev)
def kernel_circuit(x1, x2):
    feature_map(x1)
    qml.adjoint(feature_map)(x2)
    return qml.probs(wires=range(n_qubits))

# Quantum kernel function
from tqdm import tqdm

def quantum_kernel(X1, X2):
    kernel = np.zeros((len(X1), len(X2)))
    for i in tqdm(range(len(X1)), desc='Computing quantum kernel rows'):
        for j in range(len(X2)):
            kernel[i, j] = np.abs(kernel_circuit(X1[i], X2[j])[0])
    return kernel

# Step 7: Computing Quantum Kernel Matrices
print("Computing quantum kernel matrices...")
K_train = quantum_kernel(X_train, X_train)
K_test = quantum_kernel(X_test, X_train)

# Step 8: Train SVM with tho Precomputed Quantum Kernel
print("Training Quantum Kernel SVM...")
svm = SVC(kernel='precomputed')
svm.fit(K_train, y_train)

# Step 9: Evaluation
y_pred = svm.predict(K_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

print("Detailed Classification Report:")
print(classification_report(y_test, y_pred))

# Step 10: Confusion Matrix Plot
plt.figure(figsize=(12,10))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
