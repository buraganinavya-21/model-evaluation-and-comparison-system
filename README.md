# MODEL EVALUATION AND COMPARISON SYSTEM

**B.Tech Computer Science / Information Technology Capstone Project**

An interactive, end-to-end Machine Learning web application designed to evaluate, benchmark, and compare multiple classification algorithms on tabular datasets. Built with Python, Scikit-Learn, and Streamlit.

---

## 📌 1. Project Objective

The primary objective of the **Model Evaluation and Comparison System** is to provide an interactive platform for analyzing how different machine learning classification algorithms perform on the exact same dataset under identical experimental conditions.

The system helps students, researchers, and data science enthusiasts:
1. Upload custom tabular CSV datasets or test with built-in sample data.
2. Inspect dataset health, missing values, duplicates, and feature data types.
3. Perform data leakage-free preprocessing (imputation, categorical encoding, feature scaling).
4. Train 5 popular machine learning classification algorithms simultaneously.
5. Compute standard classification metrics (**Accuracy**, **Precision**, **Recall**, **F1-Score**, **ROC-AUC**).
6. Compare algorithm performances using interactive tables and Seaborn/Matplotlib visualization charts.
7. Inspect confusion matrices and classification reports for error analysis.
8. Master algorithm theory for academic Viva examinations.

---

## 🛠️ 2. Technologies Used

* **Language**: Python 3.9+
* **User Interface**: Streamlit
* **Data Processing**: Pandas, NumPy
* **Machine Learning Framework**: Scikit-Learn
* **Visualization Libraries**: Matplotlib, Seaborn

---

## 🤖 3. Machine Learning Classification Algorithms

The system implements five foundational classification algorithms:

1. **Logistic Regression**: Linear probabilistic classifier using the sigmoid activation function.
2. **K-Nearest Neighbors (KNN)**: Non-parametric, distance-based instance classifier.
3. **Decision Tree Classifier**: Non-linear tree model partitioning features using Gini Impurity / Entropy.
4. **Support Vector Machine (SVM)**: Maximum-margin boundary classifier using support vectors and kernel transformations.
5. **Random Forest Classifier**: Ensemble bagging classifier combining multiple bootstrapped decision trees.

---

## 📁 4. Project Structure

```
model-evaluation-comparison-system/
│
├── app.py                      # Main Streamlit dashboard interface
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation & Viva study guide
├── sample_dataset.csv          # Sample classification dataset (Heart Disease parameters)
│
├── src/
│   ├── __init__.py             # Python package identifier
│   ├── preprocessing.py        # Dataset stats, validation & leakage-free Scikit-Learn pipelines
│   ├── models.py               # Model instantiation & Viva educational explanations
│   ├── evaluation.py           # Model training loop & metric computation
│   └── visualization.py        # Grouped bar charts, accuracy rankings & confusion matrix heatmaps
│
└── assets/                     # Screenshots and project assets directory
```

---

## 🚀 5. Installation and Execution

### Prerequisites
Make sure you have Python (version 3.9 or higher) installed on your system.

### Step 1: Clone or Navigate to Project Folder
Open your terminal / Command Prompt / VS Code Terminal and navigate to the project directory:
```bash
cd "c:\Users\SUVARNA\Desktop\ML PROJECT"
```

### Step 2: Install Required Dependencies
Install the required packages using `pip`:
```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit Application
Launch the web interface locally:
```bash
streamlit run app.py
```
The application will automatically open in your default browser at `http://localhost:8501`.

---

## 🎓 6. Demonstrating the Project to Your Guide (Viva & Defense Guide)

Follow this step-by-step workflow during your project demo:

### Step 1: Initial Launch & Sample Dataset
* Launch the app using `streamlit run app.py`.
* Point out that the system comes pre-loaded with `sample_dataset.csv` so it works out-of-the-box.
* Show the **Dataset Overview** tab containing total row counts, column counts, missing value indicators, and target class distribution charts.

### Step 2: Data Preprocessing & Leakage Prevention
* Explain to your guide: *"To prevent data leakage, numerical feature scaling (`StandardScaler`) and categorical encoding (`OneHotEncoder`) are fit strictly on `X_train` and then applied to transform `X_test`."*
* Show the **Preprocessing Details** tab displaying the split sample counts and feature names.

### Step 3: Model Benchmark Execution
* Select all 5 algorithms in the sidebar multiselect.
* Click **"Train & Evaluate Models"**.
* Show the **Model Comparison** tab displaying the evaluation summary table. Highlight how the highest-scoring model for each metric is automatically highlighted.
* Show the comparative bar charts (Grouped Metric Comparison & Accuracy Ranking).

### Step 4: Detailed Model Inspection & Confusion Matrix
* Switch to the **Confusion Matrix** tab.
* Select different trained models from the dropdown to display their Seaborn heatmaps and full Scikit-learn Classification Reports.

### Step 5: Academic Theory (Viva Study Guide)
* Switch to the **Viva Study Guide** tab to answer any theoretical questions your guide or examiner may ask regarding algorithm working principles, hyperparameters, advantages, and limitations.

---

## 📊 7. Metric Explanations for Viva

* **Accuracy**: Proportion of correctly predicted samples out of total samples:
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
* **Precision**: Proportion of positive predictions that were actually correct:
  $$\text{Precision} = \frac{TP}{TP + FP}$$
* **Recall (Sensitivity)**: Proportion of actual positive samples correctly identified:
  $$\text{Recall} = \frac{TP}{TP + FN}$$
* **F1-Score**: Harmonic mean of Precision and Recall, ideal for imbalanced datasets:
  $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
* **ROC-AUC**: Receiver Operating Characteristic - Area Under Curve measuring binary/multiclass class separation capability across all thresholds.

---

## 🔮 8. Future Enhancements

* Add hyperparameter tuning (GridSearchCV / RandomSearchCV).
* Add automated feature selection techniques (Recursive Feature Elimination, Mutual Information).
* Add export features to download comparison reports as PDF/CSV.
* Support regression datasets in addition to classification.

---

## 📄 License & Credits
Developed as a B.Tech Capstone Project. Built using Python, Scikit-Learn, and Streamlit.
