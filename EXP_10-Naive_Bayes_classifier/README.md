Naive Bayes Classifier Implementation
===

## Introduction

This code implements a Naive Bayes classifier from scratch using Python and the pandas library. The classifier is trained on a dataset and used to make predictions on new, unseen data.

---
## Importing Libraries

```python
import pandas as pd
from collections import Counter
```

*   `pandas` is used for data manipulation and analysis.
*   `Counter` from the `collections` module is used to count the frequency of elements in a list.

---
## Naive Bayes Classifier Function

```python
def naive_bayes_classifier(X_train, y_train, X_test):
    # Calculate prior probabilities
    class_counts = Counter(y_train)
    prior_probs = {label: count / len(y_train) for label, count in class_counts.items()}

    # Calculate conditional probabilities
    conditional_probs = {}
    for feature in X_train.columns:
        conditional_probs[feature] = {}
        for class_label in y_train.unique():
            feature_values = X_train[y_train == class_label][feature]
            value_counts = Counter(feature_values)
            conditional_probs[feature][class_label] = {value: count / len(feature_values) for value, count in value_counts.items()}
    print(conditional_probs)
    print('\n\n')
    # Make predictions
    predictions = []
    for x in X_test.to_dict(orient='records'):
        probabilities = {}
        for class_label in y_train.unique():
            probability = prior_probs[class_label]
            for feature, value in x.items():
                if feature in conditional_probs and value in conditional_probs[feature][class_label]:
                    probability *= conditional_probs[feature][class_label][value]
                else:
                    probability *= 1e-9  # Handle unseen values
            probabilities[class_label] = probability
        predicted_class = max(probabilities, key=probabilities.get)
        predictions.append(predicted_class)

    return predictions
```

#

```python
def naive_bayes_classifier(X_train, y_train, X_test):
```

This function takes in three parameters:
*   `X_train`: The training features.
*   `y_train`: The training target variable.
*   `X_test`: The testing features.


### Calculating Prior Probabilities

```python
class_counts = Counter(y_train)
prior_probs = {label: count / len(y_train) for label, count in class_counts.items()}
```
This code calculates the prior probabilities of each class label in the training data. The `Counter` class from the `collections` module is used to count the occurrences of each class label. The prior probabilities are then calculated by dividing the count of each class label by the total number of samples in the training data.

The prior probabilities are calculated using the formula:

$$P(c) = \frac{Count(c)}{Total Count}$$

where *c* is a class label.

#

### Calculating Conditional Probabilities

```python
conditional_probs = {}
for feature in X_train.columns:
    conditional_probs[feature] = {}
    for class_label in y_train.unique():
        feature_values = X_train[y_train == class_label][feature]
        value_counts = Counter(feature_values)
        conditional_probs[feature][class_label] = {value: count / len(feature_values) for value, count in value_counts.items()}
```

This code calculates the conditional probabilities of each feature value given a class label. The `Counter` class from the `collections` module is used to count the occurrences of each feature value. The conditional probabilities are then calculated by dividing the count of each feature value by the total number of samples in the training data for the given class label.

`conditional_probs = {}` : This is an empty dictionary that will store the conditional probabilities of each feature value given a class label.

```python
for feature in X_train.columns:
    conditional_probs[feature] = {}
```
This loop iterates over each feature in the training data 

`conditional_probs[feature] = {}` : This line creates an empty dictionary for each feature that will store the conditional probabilities of each feature value given a class label.

```python
    for class_label in y_train.unique():
            feature_values = X_train[y_train == class_label][feature]
            value_counts = Counter(feature_values)
            conditional_probs[feature][class_label] = {value: count / len(feature_values) for value, count in value_counts.items()}
```
This nested loop iterates over each unique class label in the training data. For each class label, it selects the feature values for that class label, counts the occurrences of each feature value, and calculates the conditional probability of each feature value given the class label.

`feature_values = X_train[y_train == class_label][feature]`
 - `X_train` : This is the training data features.
 - `y_train == class_label` : This is a boolean mask that selects the rows in X_train where the class label matches the current class label being processed.
 - `[feature]`: This is used to select a specific feature from the filtered data.

 So, the line is essentially selecting the values of a specific feature for a specific class label from the training data.


The conditional probabilities are calculated using the formula:

$$P(x|c) = \frac{Count(x, c)}{Count(c)}$$

where *x* is a feature value and *c* is a class label.


### Making Predictions

```python
predictions = []
for x in X_test.to_dict(orient='records'):
    probabilities = {}
    for class_label in y_train.unique():
        probability = prior_probs[class_label]
        for feature, value in x.items():
            if feature in conditional_probs and value in conditional_probs[feature][class_label]:
                probability *= conditional_probs[feature][class_label][value]
            else:
                probability *= 1e-9  # Handle unseen values
        probabilities[class_label] = probability
    predicted_class = max(probabilities, key=probabilities.get)
    predictions.append(predicted_class)
```
This code makes predictions on the test data using the Bayes' theorem. It iterates over each test sample, calculates the probability of each class label given the sample, and selects the class label with the highest probability as the predicted class.

The predictions are made using the formula:


$$P(c|x) = \frac{P(x|c)  P(c)}{P(x)}$$

where *c* is a class label and *x* is a feature vector. The class label with the highest posterior probability is selected as the predicted class.

---
## Loading Data

```python
data = pd.read_csv("data.csv")
```

The dataset is loaded from a CSV file using pandas.

---
## Separating Features and Target Variable

```python
X = data.drop('buys_computer', axis=1)
y = data['buys_computer']
```

The features and target variable are separated.

---
## Predicting the Class Label for a Given Tuple

```python
tuple_to_classify = {'age': 'youth', 'income': 'medium', 'student': 'yes', 'credit': 'fair'}
predicted_class = naive_bayes_classifier(X_train, y_train, pd.DataFrame([tuple_to_classify]))
print("Predicted class:", predicted_class)
```

The classifier is used to predict the class label for a given tuple.
  
Complete Code
--
```python
import pandas as pd
from collections import Counter

def naive_bayes_classifier(X_train, y_train, X_test):
    # Calculate prior probabilities
    class_counts = Counter(y_train)
    prior_probs = {label: count / len(y_train) for label, count in class_counts.items()}

    # Calculate conditional probabilities
    conditional_probs = {}
    for feature in X_train.columns:
        conditional_probs[feature] = {}
        for class_label in y_train.unique():
            feature_values = X_train[y_train == class_label][feature]
            value_counts = Counter(feature_values)
            conditional_probs[feature][class_label] = {value: count / len(feature_values) for value, count in value_counts.items()}
    print(conditional_probs)
    print('\n\n')
    # Make predictions
    predictions = []
    for x in X_test.to_dict(orient='records'):
        probabilities = {}
        for class_label in y_train.unique():
            probability = prior_probs[class_label]
            for feature, value in x.items():
                if feature in conditional_probs and value in conditional_probs[feature][class_label]:
                    probability *= conditional_probs[feature][class_label][value]
                else:
                    probability *= 1e-9  # Handle unseen values
            probabilities[class_label] = probability
        predicted_class = max(probabilities, key=probabilities.get)
        predictions.append(predicted_class)

    return predictions

# Load data from CSV
data = pd.read_csv("data.csv")

# Separate features and target variable
X = data.drop('buys_computer', axis=1)
y = data['buys_computer']


# Predict the class label for the given tuple
tuple_to_classify = {'age': 'youth', 'income': 'medium', 'student': 'yes', 'credit': 'fair'}
predicted_class = naive_bayes_classifier(X,y, pd.DataFrame([tuple_to_classify]))
print("Predicted class:", predicted_class)
```