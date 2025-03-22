# Titanic-Survival-Prediction
## Project Overview
This machine learning project aims to predict the survival status of passengers aboard the Titanic based on various features such as age, gender, ticket class, fare, cabin information, etc. The goal is to develop a well-trained classification model that can accurately predict whether a passenger survived the Titanic disaster.

## Dataset Link: https://www.kaggle.com/datasets/brendan45774/test-file

## Key Features:
Age: The age of the passenger.

Gender: The gender of the passenger.

Pclass: The passenger's ticket class (1st, 2nd, or 3rd).

Fare: The fare paid by the passenger for the ticket.

SibSp: The number of siblings/spouses aboard.

Parch: The number of parents/children aboard.

Embarked: The port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton).

## Expected Outcome:
A well-trained classification model capable of accurately predicting whether a passenger survived or not.

## Installation
To get started with the Titanic Survival Prediction project, follow these steps:

1. Clone the repository to your local machine:

git clone https://github.com/Khushii2308/Titanic-Survival-Prediction.git

2. Navigate into the project directory:

cd Titanic-Survival-Prediction

3. Install the required dependencies:


pip install -r requirements.txt

## Data Preprocessing
The dataset contains missing values and categorical variables that need to be handled before building the machine learning model. The preprocessing steps include:

Handling Missing Values: We impute missing values for columns such as Age, Embarked, and Cabin.

Encoding Categorical Variables: We use techniques like One-Hot Encoding and Label Encoding to convert categorical features (e.g., Gender, Embarked) into numerical format.

Feature Scaling: We normalize numerical features like Age, Fare, SibSp, and Parch to ensure that the model can perform optimally.

Feature Engineering: New features, such as family size (SibSp + Parch), are created to potentially enhance the model's performance.

## Model Training
The project explores several machine learning models, including but not limited to:

Logistic Regression: A simple but effective classification model.

Random Forest: A more complex model for capturing non-linear relationships.

Support Vector Machine (SVM): A powerful classifier for binary outcomes.

K-Nearest Neighbors (KNN): A non-parametric method for classification.

Each model is trained on the training dataset, and hyperparameters are tuned for optimal performance.

## Model Evaluation

Model performance is evaluated using multiple metrics:

1. Accuracy: The percentage of correct predictions out of total predictions.

2. Precision: The proportion of true positives out of all positive predictions.

3. Recall: The proportion of true positives out of all actual positives.

4. F1-Score: The harmonic mean of precision and recall.

5. ROC-AUC: The area under the Receiver Operating Characteristic curve.

6. We compare the models based on these metrics and select the one with the best performance on the validation set.

## Results
The final model's performance is measured on the test dataset. The model's accuracy and other evaluation metrics are included in the results section.

Best Model: Random Forest achieved the highest accuracy of 85% with a strong balance of precision and recall.

## Usage
After training the model, you can predict the survival status of new passengers by loading the trained model and passing in the required features.

Example usage in Python:

from src.model import load_model, make_predictions

### Load the trained model
model = load_model('outputs/model.pkl')

### Example passenger data
passenger_data = {
    'Pclass': 1,
    'Sex': 'female',
    'Age': 22,
    'SibSp': 1,
    'Parch': 0,
    'Fare': 71.2833,
    'Embarked': 'C'
}

### Make prediction
survival_prediction = make_predictions(model, passenger_data)
print(f"Survived: {survival_prediction}")

## Contribution
Feel free to fork the repository, improve the model, and submit pull requests. Contributions such as adding new models, improving data preprocessing, or optimizing the training process are always welcome.

## Acknowledgments
The Titanic dataset is provided by Kaggle.

The project uses Scikit-learn for machine learning algorithms and Pandas for data manipulation.

