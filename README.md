# 🧠 AI-Powered Disease Prediction from Custom CSV Data

This project is a machine learning-based disease prediction system that allows users to upload **custom health datasets** (in CSV format) via **Google Colab**. It trains a **Random Forest Classifier** to predict diseases based on symptoms, providing insights into the most important features influencing predictions.

---

## 📂 File Overview

- `disease_prediction_custom.py`: Main Python script to load, preprocess, train, and evaluate a disease prediction model using a user-uploaded dataset.

---

## 🚀 How to Run the Project

### 🔗 Open with Google Colab

> Click the button below to run directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

### 📌 Steps to Execute

1. **Upload Your CSV Dataset**

   - Run the script in Google Colab.
   - A file upload prompt will appear.
   - Upload your CSV file (with symptom columns and a target column such as `prognosis` or `diagnosis`).

2. **Dataset Loading & Validation**

   - The script reads the uploaded CSV using `pandas`.
   - If the `prognosis` column is not found, you’ll be prompted to enter the correct column name for the disease label.

3. **Data Preprocessing**

   - Removes missing values and duplicate rows.
   - Encodes the target column using `LabelEncoder`.
   - One-hot encodes the input symptom features using `OneHotEncoder`.

4. **Model Training**

   - The data is split into training and test sets (80% training, 20% test).
   - A `RandomForestClassifier` is trained on the training data.

5. **Model Evaluation**

   - Displays:
     - **Accuracy Score**
     - **Classification Report** (Precision, Recall, F1-score)
     - **Confusion Matrix** using Seaborn
     - **Top 10 Feature Importance** plot showing which symptoms contributed most to the prediction.

---

## 🧪 Example Output

### ✅ Accuracy

Accuracy: 0.97



### 📊 Classification Report


          precision    recall  f1-score   support
 Disease1       0.97      0.96      0.96        50
 Disease2       0.98      0.99      0.98        60






## 📄 Dataset Format Example

Your input dataset should be in CSV format with symptoms as features and a disease label as the target column.

| fever | cough | sore_throat | nausea | prognosis |
|-------|-------|-------------|--------|-----------|
| yes   | no    | yes         | no     | Flu       |
| no    | yes   | no          | yes    | Typhoid   |

> Target column must be named `prognosis`, or you'll be prompted to rename it during runtime.

---

## 🛠 Dependencies

Install the required Python libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

.
├── disease_prediction_custom.py
├── README.md
├── sample_dataset.csv           # (optional: upload your test dataset)
├── assets/
│   ├── confusion_matrix.png
│   └── feature_importance.png


🤖 Model Details
Model: Random Forest Classifier

n_estimators: 100

Random State: 42

Suitable for categorical and numerical inputs (via one-hot encoding).

Robust to overfitting and interpretable through feature importance.

✨ Future Improvements
Add a form-based UI for live symptom input.

Deploy using Flask or Streamlit for web use.

Integrate with medical APIs or hospital data sources.

Add real-time prediction via web or mobile app.
