### 📊 Dataset
Source: (Heart Disease Dataset)[https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data]

Description: The dataset includes various clinical features such as age, cholesterol levels, blood pressure, and more, aimed at determining the presence of heart disease.

### 🚀 Features
#### Data Preprocessing:
Handling missing values using different methods like iterative imputer or filling by constant value, encoding categorical variables, and feature scaling.

#### Feature Selection:
Application of Recursive Feature Elimination (RFE) and Chi-Square tests to identify significant predictors. I also plotted each feature importance.

#### Dimensionality Reduction:
Used PCA to preserve the most variance for Unsupervised Machine learning models.

#### Model Training:
##### Supervised:
Implementation of multiple classifiers including Logistic Regression, Random Forest, Decision Tree, and Support Vector Machine (SVM).

##### UnSupervised:
Kmeans and Hierarchical Clustering to group similar objects.

#### Hyperparameter Tuning:
Used GridSearch and RandomizedSearch to get the best paramters for models.

#### Model Evaluation:
Utilization of metrics like ROC curves and AUC scores to assess model performance.

#### Model Persistence:
Saving trained models using joblib for future inference.

### 🛠️ Installation
1. Clone the repository:
```bash
git clone https://github.com/yahia997/SprintUp-AI-ML-project.git
cd SprintUp-AI-ML-project
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```