##  Car Evaluation Classification

In this project, we aimed to classify car evaluations using two popular machine learning models: Random Forest Classifier and XGBoost Random Forest Classifier. The process involved several steps including data preprocessing, model training, hyperparameter tuning, and evaluation.

### Dataset

The dataset used for this project is the "Car Evaluation" dataset. It contains categorical features related to car attributes and a target variable representing the evaluation of the car. The columns in the dataset are:
- **buying**: buying price
- **meant**: maintenance price
- **doors**: number of doors
- **persons**: capacity in terms of persons to carry
- **lug_boot**: the size of luggage boot
- **safety**: estimated safety of the car
- **class**: car evaluation (target variable)

### Data Preprocessing

1. **Data Loading**: The dataset was loaded using `pandas`.
2. **Data Splitting**: The data was split into training and testing sets using `train_test_split` from `scikit-learn`.
3. **Data Encoding**: Categorical features were encoded using `OrdinalEncoder` from the `category_encoders` library.

### Model Training and Evaluation

Two models were trained and evaluated:

1. **Random Forest Classifier**:
   - Initial model training and evaluation.
   - Hyperparameter tuning using `GridSearchCV` to find the best parameters.
   - Evaluation using accuracy, confusion matrix, ROC curve, and classification report.

2. **XGBoost Random Forest Classifier**:
   - Model training and evaluation with hyperparameter tuning using `GridSearchCV`.
   - Evaluation using the same metrics as the Random Forest Classifier.

### Evaluation Metrics

The models were evaluated using:
- **Accuracy**: Overall accuracy of the model.
- **Confusion Matrix**: To visualize the performance of the classification model.
- **ROC Curve and AUC**: To assess the model's ability to distinguish between classes.
- **Classification Report**: To provide precision, recall, and F1-score for each class.

### Results

- The Random Forest Classifier with tuned hyperparameters performed slightly better than the default Random Forest.
- The XGBoost Random Forest Classifier performed slightly worse than the Random Forest Classifier.

### Technologies Used

- **Programming Language**: Python
- **Data Manipulation and Analysis**: `pandas`, `numpy`
- **Model Training and Evaluation**: `scikit-learn`
- **Hyperparameter Tuning**: `GridSearchCV`
- **Visualization**: `seaborn`, `matplotlib`
- **Model**: `RandomForestClassifier` from `scikit-learn`, `XGBRFClassifier` from `xgboost`
- **Encoding**: `category_encoders`

This project demonstrates the importance of data preprocessing, model selection, hyperparameter tuning, and thorough evaluation in machine learning applications.
