# disease_prediction_custom.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
from google.colab import files

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 required for 3D plots

# Upload the dataset using Google Colab's file upload feature
uploaded = files.upload()

# Get the first filename from the uploaded dictionary
filename = list(uploaded.keys())[0]  # Assuming only one file is uploaded

# Read the uploaded file into a pandas DataFrame
try:
    df = pd.read_csv(io.BytesIO(uploaded[filename]))  # Use the actual filename
    print("File loaded successfully.")
except KeyError:
    print(f"File '{filename}' not found in uploaded files.")
    raise

# Display basic info
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Preprocessing
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Check if 'prognosis' column exists
if 'prognosis' not in df.columns:
    print("Error: 'prognosis' column not found in the dataset. Please check your CSV file.")
    target_column = input("Enter the name of the target column: ")
    df.rename(columns={target_column: 'prognosis'}, inplace=True)
else:
    print("'prognosis' column found in the dataset.")

# Encode the target column
label_encoder = LabelEncoder()
df['prognosis'] = label_encoder.fit_transform(df['prognosis'])

# Split features and target
X = df.drop('prognosis', axis=1)

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_data = encoder.fit_transform(X)
encoded_feature_names = encoder.get_feature_names_out(X.columns)
X = pd.DataFrame(encoded_data, columns=encoded_feature_names)

y = df['prognosis']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=False, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature importance - Bar chart
importances = model.feature_importances_
indices = np.argsort(importances)[-10:]  # Top 10 important features

plt.figure(figsize=(10, 6))
plt.title("Top 10 Important Symptoms")
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel("Feature Importance")
plt.show()

# 3D PCA plot
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                     c=y, cmap='rainbow', edgecolor='k', s=40)
ax.set_title("3D PCA Projection of Disease Data")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")
legend = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend)
plt.show()
