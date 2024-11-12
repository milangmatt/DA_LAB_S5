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
    filename = 'EXP_11-Decision_tree/data.csv'  # Name of your CSV file
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
