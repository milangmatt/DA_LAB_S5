Decision Tree 
=====================================

## Introduction

This code implements a decision tree algorithm in Python. It reads a CSV file, calculates the entropy of the dataset, splits the dataset based on the best feature, and builds a decision tree recursively. The decision tree is then used to classify new samples.

---


## Read CSV Data

This function reads a CSV file and returns a list of tuples, where each tuple contains the features and the label.

```python
def read_csv_file(filename):
    data = []
    with open(filename, 'r') as file:
        next(file)  # Skip the header row
        for line in file:
            row = line.strip().split(',')
            features = row[1:-1]  # Age, income, student, credit_rating
            label = row[-1]       # buys_computer
            data.append((features, label))
    return data
```
Each line in the file is stripped of the white spaces and split on ( , ) and is made into a list with two elements. The first element is a list of the features and the second element is the label.

---

## Calculate Entropy of a Dataset

This function calculates the entropy of a dataset using the Shannon entropy formula:

$$H(X) = - ∑ (p(x) * log2(p(x)))$$


where p(x) is the probability of each label.

```python
def entropy(data):
    total_samples = len(data) # number of rows
    label_counts = {} # to count number of individual label

    for features, label in data:
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

    ent = 0
    for label in label_counts:
        prob = label_counts[label] / total_samples
        ent -= prob * math.log2(prob)

    return ent
```
This function first counts the number of each label in the dataset. Then it calculates the probability of each label by dividing the count of each label by the total number of samples. Finally, it calculates the entropy using the Shannon entropy formula.

The input data would always be changing as the tree is split at every split point and changed to true branch and false branch.

```python
    total_samples = len(data) # number of rows
    label_counts = {} # to count number of individual label
```
Empty Dictionary to store the count of labels is initialized.

```python
    for features, label in data:
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1
```
The count for each label is recorded.

```python
    ent = 0
    for label in label_counts:
        prob = label_counts[label] / total_samples
        ent -= prob * math.log2(prob)
```
The probability of each label is calculated and the entropy is calculated using the Shannon entropy formula.


---

## Split Dataset Based on a Feature and its Value

This function splits the dataset into two branches based on a feature and its value.

```python
def split_data(data, index, value):
    true_branch = [row for row in data if row[0][index] == value]
    false_branch = [row for row in data if row[0][index] != value]
    return true_branch, false_branch
```

This function takes in the dataset, the index of the feature to split on, and the value to split on. It then uses list comprehension to create two lists: one for the rows where the feature at the given index has the given value, and one for the rows where it does not.

`[row for row in data if row[0][index] == value]`

This line of code is used to create a new list that includes only the rows where the feature at the given index has the given value.


---

## Find the Best Feature to Split on Using Information Gain

This function finds the best feature to split on using the information gain formula:

$$IG(X, Y) = H(X) - H(X|Y)$$

Total gain is Given by :

$$gain = H(X) - (p * H(X|Y=true) + (1-p) * H(X|Y=false))$$

where H(X) is the entropy of the dataset, and H(X|Y) is the conditional entropy of the dataset given the feature Y.

```python
def find_best_split(data):
    best_gain = 0
    best_index = None
    best_value = None
    current_entropy = entropy(data)
    n_features = len(data[0][0])  # Number of features

    for index in range(n_features):
        values = set([row[0][index] for row in data])  # Unique values in the column
        for value in values:
            true_branch, false_branch = split_data(data, index, value)

            if not true_branch or not false_branch:
                continue

            # Calculate information gain
            p = len(true_branch) / len(data)
            gain = current_entropy - p * entropy(true_branch) - (1 - p) * entropy(false_branch)

            if gain > best_gain:
                best_gain, best_index, best_value = gain, index, value

    return best_gain, best_index, best_value
```
This function iterates over each feature in the dataset, and for each feature, it iterates over each unique value in that feature. It then uses the `split_data` function to split the data into two branches based on the current feature and value. It calculates the information gain by using the `entropy` function to calculate the entropy of the dataset and the two branches.

```python
    best_gain = 0
    best_index = None
    best_value = None
    current_entropy = entropy(data)
    n_features = len(data[0][0])
```
This code initializes the best gain, best index, and best value to 0, None, and None respectively. It also calculates the current entropy of the dataset.

```python
    for index in range(n_features):
        values = set([row[0][index] for row in data])
```
This code iterates over each feature in the dataset and creates a set of unique values in that feature.

```python
        for value in values:
            true_branch, false_branch = split_data(data, index, value)

            if not true_branch or not false_branch:
                continue
```
This code iterates over each unique value in the feature and splits the data into two branches based on the current feature and value. If either branch is empty, it skips to the next value.

```python
            p = len(true_branch) / len(data)
            gain = current_entropy - p * entropy(true_branch) - (1 - p) * entropy(false_branch)
```
This code calculates the probability of the true branch and the information gain by using the `entropy` function.

```python
            if gain > best_gain:
                best_gain, best_index, best_value = gain, index, value
```
This code checks if the current gain is greater than the best gain found so far, and if so updates the best gain, best index, and best value.



---

## Decision Node Class

```python
class DecisionNode:
    def __init__(self, index=None, value=None, true_branch=None, false_branch=None, prediction=None):
        self.index = index
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.prediction = prediction
```
This class represents a decision node in the decision tree. It has attributes for the index of the feature used to split the data, the value of the feature used to split the data, the true branch and false branch of the node, and the prediction made by the node.


## Build the Decision Tree Recursively

This function builds the decision tree recursively by finding the best feature to split on and splitting the dataset accordingly.

```python
def build_tree(data):
    gain, index, value = find_best_split(data)

    if gain == 0:  # No further gain, return a leaf node
        return DecisionNode(prediction=data[0][1])

    true_branch, false_branch = split_data(data, index, value)
    true_node = build_tree(true_branch)
    false_node = build_tree(false_branch)

    return DecisionNode(index=index, value=value, true_branch=true_node, false_branch=false_node)
```
This function takes in the dataset and recursively builds the decision tree by finding the best feature to split on and splitting the dataset accordingly. If there is no further gain in splitting the data, it returns a leaf node with the prediction made by the majority vote of the data points in the dataset.

`gain, index, value = find_best_split(data)` : This line finds the best feature to split on by calling the `find_best_split` function.

```python
    if gain == 0:  # No further gain, return a leaf node
        return DecisionNode(prediction=data[0][1])
```
This line checks if there is no further gain in splitting the data. If so, it returns a leaf node with the prediction made by the majority vote of the data points in the dataset.

```python
true_branch, false_branch = split_data(data, index, value)
    true_node = build_tree(true_branch)
    false_node = build_tree(false_branch)

    return DecisionNode(index=index, value=value, true_branch=true_node, false_branch=false_node)
```
This block of code splits the dataset into two branches based on the best feature to split on and recursively builds the decision tree for each branch.

---

## Print the Decision Tree

This function prints the decision tree in a readable format.

```python
def print_tree(node, headers, spacing="", level=0):
    if node.prediction is not None:
        print(spacing*2 + f"Predict: {node.prediction}")
        return

    print(f" {spacing}{spacing} |( {headers[node.index]} == {node.value} ) ?")
    print(spacing + '  ' * level + '  |')
    print(spacing + '  ' * level + '  |--> True:')
    print_tree(node.true_branch, headers, spacing + '  ', level + 1)
    print(spacing + '  ' * level + '  |--> False:')
    print_tree(node.false_branch, headers, spacing + '  ', level + 1)
```
This function takes in the decision tree node, the headers of the dataset, and the current spacing and level of indentation. It prints the decision tree in a readable format by recursively calling itself for the true and false branches of the node.

```python
    if node.prediction is not None:
        print(spacing*2 + f"Predict: {node.prediction}")
        return
```
This line checks if the node is a leaf node. If so, it prints the prediction made by the majority vote of the data points in the dataset.


---

## Classify a New Sample Using the Decision Tree

This function classifies a new sample using the decision tree.

```python
def classify(tree, sample):
    if tree.prediction is not None:
        return tree.prediction

    if sample[tree.index] == tree.value:
        return classify(tree.true_branch, sample)
    else:
        return classify(tree.false_branch, sample)
```
This function takes in the decision tree and a sample to classify. It recursively traverses the decision tree based on the feature values of the sample and returns the predicted class.

```python
    if tree.prediction is not None:
        return tree.prediction
```
This line checks if the node is a leaf node. If so, it returns the prediction made by the majority vote of the data points in the dataset.

```python
    if sample[tree.index] == tree.value:
        return classify(tree.true_branch, sample)
    else:
        return classify(tree.false_branch, sample)
```
This code recursively traverses the decision tree based on the feature values of the sample.



---

## Main Execution

```python
if __name__ == "__main__":
    filename = 'data.csv'  # Name of your CSV file
    headers = ['age', 'income', 'student', 'credit_rating']  # Feature names
    training_data = read_csv_file(filename)

    # Build the decision tree
    tree = build_tree(training_data)

    # Print the decision tree
    print("Decision Tree Structure:")
    print_tree(tree, headers)


    print("Select features for new sample:")
    age = input("Enter age (youth,middle_aged, senior): ")
    income = input("Enter income (low, medium, high): ")
    student = input("Enter student (yes, no): ")
    credit_rating = input("Enter credit rating (fair, good, excellent): ")
    new_sample = [age, income, student, credit_rating]
    predicted_class = classify(tree, new_sample)
    print(f'\nPredicted class for {new_sample}: {predicted_class}')
```
This code will read the CSV file, build the decision tree, print the tree structure, and classify a new sample based on the user's input. The predicted class will be printed to the console.

## Complete Code

```python
import csv
import math

# Step 1: Read CSV Data
def read_csv_file(filename):
    data = []
    with open(filename, 'r') as file:
        next(file)  # Skip the header row
        for line in file:
            row = line.strip().split(',')
            features = row[1:-1]  # Age, income, student, credit_rating
            label = row[-1]       # buys_computer
            data.append((features, label))
    return data

# Step 2: Calculate Entropy of a dataset
def entropy(data):
    total_samples = len(data) # number of rows
    label_counts = {} # to count number of individual label

    for features, label in data:
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

    ent = 0
    for label in label_counts:
        prob = label_counts[label] / total_samples
        ent -= prob * math.log2(prob)

    return ent

# Step 3: Split dataset based on a feature and its value
def split_data(data, index, value):
    true_branch = [row for row in data if row[0][index] == value]
    false_branch = [row for row in data if row[0][index] != value]
    return true_branch, false_branch

# Step 4: Find the best feature to split on using Information Gain
def find_best_split(data):
    best_gain = 0
    best_index = None
    best_value = None
    current_entropy = entropy(data)
    n_features = len(data[0][0])  # Number of features

    for index in range(n_features):
        values = set([row[0][index] for row in data])  # Unique values in the column
        for value in values:
            true_branch, false_branch = split_data(data, index, value)

            if not true_branch or not false_branch:
                continue

            # Calculate information gain
            p = len(true_branch) / len(data)
            gain = current_entropy - p * entropy(true_branch) - (1 - p) * entropy(false_branch)

            if gain > best_gain:
                best_gain, best_index, best_value = gain, index, value

    return best_gain, best_index, best_value

# Step 5: Build the Decision Tree recursively
class DecisionNode:
    def __init__(self, index=None, value=None, true_branch=None, false_branch=None, prediction=None):
        self.index = index
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.prediction = prediction

def build_tree(data):
    gain, index, value = find_best_split(data)

    if gain == 0:  # No further gain, return a leaf node
        return DecisionNode(prediction=data[0][1])

    true_branch, false_branch = split_data(data, index, value)
    true_node = build_tree(true_branch)
    false_node = build_tree(false_branch)

    return DecisionNode(index=index, value=value, true_branch=true_node, false_branch=false_node)

# Step 6: Print the Decision Tree
def print_tree(node, headers, spacing="", level=0):
    if node.prediction is not None:
        print(spacing*2 + f"Predict: {node.prediction}")
        return

    print(f" {spacing}{spacing} |( {headers[node.index]} == {node.value} ) ?")
    print(spacing + '  ' * level + '  |')
    print(spacing + '  ' * level + '  |--> True:')
    print_tree(node.true_branch, headers, spacing + '  ', level + 1)
    print(spacing + '  ' * level + '  |--> False:')
    print_tree(node.false_branch, headers, spacing + '  ', level + 1)

# Step 7: Classify a new sample using the Decision Tree
def classify(tree, sample):
    if tree.prediction is not None:
        return tree.prediction

    if sample[tree.index] == tree.value:
        return classify(tree.true_branch, sample)
    else:
        return classify(tree.false_branch, sample)

# Main execution
if __name__ == "__main__":
    filename = 'data.csv'  # Name of your CSV file
    headers = ['age', 'income', 'student', 'credit_rating']  # Feature names
    training_data = read_csv_file(filename)

    # Build the decision tree
    tree = build_tree(training_data)

    # Print the decision tree
    print("Decision Tree Structure:")
    print_tree(tree, headers)

    # Classify a new sample: X = (age=youth, income=medium, student=yes, credit_rating=fair)
    print("Select features for new sample:")
    age = input("Enter age (youth,middle_aged, senior): ")
    income = input("Enter income (low, medium, high): ")
    student = input("Enter student (yes, no): ")
    credit_rating = input("Enter credit rating (fair, good, excellent): ")
    new_sample = [age, income, student, credit_rating]
    predicted_class = classify(tree, new_sample)
    print(f'\nPredicted class for {new_sample}: {predicted_class}')
```
