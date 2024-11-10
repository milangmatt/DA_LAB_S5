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

# Preprocess data (handle missing values, encode categorical features, etc.)
# ...

# Separate features and target variable
X = data.drop('buys_computer', axis=1)
y = data['buys_computer']

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier
predictions = naive_bayes_classifier(X_train, y_train, X_test)


# Evaluate the classifier
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Predict the class label for the given tuple
tuple_to_classify = {'age': 'youth', 'income': 'medium', 'student': 'yes', 'credit': 'fair'}
predicted_class = naive_bayes_classifier(X_train, y_train, pd.DataFrame([tuple_to_classify]))
print("Predicted class:", predicted_class)