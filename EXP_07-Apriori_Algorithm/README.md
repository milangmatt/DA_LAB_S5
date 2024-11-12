Apriori Algorithm
================


## Introduction

This code is an implementation of the Apriori algorithm, a popular algorithm in data mining for finding frequent itemsets in a dataset. The code is written in Python and uses the csv library to read data from a csv file.

## Importing Libraries

```python
import csv
from itertools import combinations
from collections import defaultdict
```

These libraries are used for the following purposes:

*   csv: to read data from a csv file
*   itertools: to generate combinations of items
*   collections: to use the defaultdict data structure
  
`defaultdict` : A dictionary that provides a default value for the key that does not exist.

---
## Function: get_frequent_itemsets

```python
def get_frequent_itemsets(transactions, min_support):
    itemsets = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            itemsets[frozenset([item])] += 1
    itemsets = {itemset: count for itemset, count in itemsets.items() if count >= min_support}
    return itemsets
```

This function takes a list of transactions and a minimum support threshold as input. It returns a dictionary of frequent itemsets, where each itemset is a frozenset of items and the value is the count of the itemset in the transactions.

```python
def get_frequent_itemsets(transactions, min_support):
```
This function takes two parameters:
- `transactions`: a list of transactions, where each transaction is a list of items.
- `min_support`: the minimum support threshold, which is the minimum number of transactions that an item

```python
    itemsets = defaultdict(int)
```
This line initializes a dictionary called `itemsets` with default value 0.

```python
    for transaction in transactions:
        for item in transaction:
            itemsets[frozenset([item])] += 1
```
This loop iterates over each transaction and each item in the transaction. For each item, it increments the count of the itemset containing that item in the `itemsets` dictionary.

```python
     itemsets = {itemset: count for itemset, count in itemsets.items() if count >= min_support}
```
This line filters the `itemsets` dictionary to only include itemsets with a count greater than or equal to the minimum support threshold.

```python
    return itemsets
```
This line returns the filtered `itemsets` dictionary.

---
## Function: generate_candidates

```python
def generate_candidates(prev_frequent_itemsets, k):
    candidates = set()
    itemsets_list = list(prev_frequent_itemsets)
    for i in range(len(itemsets_list)):
        for j in range(i + 1, len(itemsets_list)):
            candidate = itemsets_list[i] | itemsets_list[j]
            if len(candidate) == k:
                subsets = list(combinations(candidate, k - 1))
                if all(frozenset(subset) in prev_frequent_itemsets for subset in subsets):
                    candidates.add(candidate)
    return candidates
```

This function takes a dictionary of previous frequent itemsets and a value k as input. It returns a set of candidate itemsets of size k.

```python
def generate_candidates(prev_frequent_itemsets, k):
```
This line defines the function `generate_candidates` that takes two parameters:
- `prev_frequent_itemsets` : a dictionary of previous frequent itemsets.
- `k` : desired size of candidate sets


```python
    candidates = set()
    itemsets_list = list(prev_frequent_itemsets)
```
This line initializes an empty set `candidates` to store the generated candidate itemsets. It also converts the dictionary `prev_frequent_itemsets` into a list of itemsets for easier iteration.


```python
    for i in range(len(itemsets_list)):
        for j in range(i + 1, len(itemsets_list)):
```
This nested loop iterates over all pairs of itemsets in `itemsets_list`. The outer loop iterates over each itemset, and the inner loop iterates over the itemsets that come after it in the list. This is so as to ensure no repetition is occuring in candidate sets.


```python
           candidate = itemsets_list[i] | itemsets_list[j] 
```
This line generates a candidate itemset by taking the union of the two itemsets being compared.

```python
           if len(candidate) == k:
                subsets = list(combinations(candidate, k - 1))
                if all(frozenset(subset) in prev_frequent_itemsets for subset in subsets):
                    candidates.add(candidate)
```
This block of code checks if the size of the candidate itemset is equal to k. If it is, it generates all possible subsets of size k-1 from the candidate itemset. 

It then checks if all these subsets are present in the previous frequent itemsets. If they are, the candidate itemset is added to the set of candidates.


```python
    return candidates
```
This line returns the set of candidate itemsets.

---
## Function: apriori

```python
def apriori(transactions, min_support):
    transactions = list(map(set, transactions))
    itemsets = get_frequent_itemsets(transactions, min_support)
    all_frequent_itemsets = dict(itemsets)

    k = 2
    while itemsets:
        candidates = generate_candidates(itemsets, k)
        itemsets = defaultdict(int)

        for transaction in transactions:
            for candidate in candidates:
                if candidate.issubset(transaction):
                    itemsets[candidate] += 1
        itemsets = {itemset: count for itemset, count in itemsets.items() if count >= min_support}
        all_frequent_itemsets.update(itemsets)
        k += 1

    return all_frequent_itemsets
```

This function takes a list of transactions and a minimum support threshold as input. It returns a dictionary of all frequent itemsets.


```python
def apriori(transactions, min_support):
```
This line defines the function apriori, which takes two parameters:
- `transactions` : a list of transactions, where each transaction is a set of items.
- `min_support` : the minimum support threshold, which is the minimum number of transactions that an itemset


```python
    transactions = list(map(set, transactions))
    itemsets = get_frequent_itemsets(transactions, min_support)
    all_frequent_itemsets = dict(itemsets)
```
This block of code converts the transactions into sets of items, gets the frequent itemsets with the minimum support, and stores them in a dictionary called `all_frequent_itemsets`.

The frequent itemset is generated only once as it creates frequent 1-itemsets that has minimum support.


```python
    k = 2
```
This line initializes a variable `k` to 2, which will be used to keep track of the size of the itemsets.


```python
    while itemsets:
        candidates = generate_candidates(itemsets, k)
        itemsets = defaultdict(int)
```
This block of code generates candidates for the current size `k` and initializes an empty dictionary `itemsets` to store the frequent itemsets for the next size `k+1`.

The size of itemsets in `itemsets` will be k-1 and the size of itemsets in `candidates` will be k. 

The `generate_candidates` function generates all possible combinations of size `k` from the frequent itemsets of size `k-1` and returns them as a set of sets.


```python
        for transaction in transactions:
                for candidate in candidates:
                    if candidate.issubset(transaction):
                        itemsets[candidate] += 1
```
This block of code iterates over each transaction and each candidate. If a candidate is a subset of a transaction, it increments the count of the candidate in the `itemsets` dictionary.


```python
        itemsets = {itemset: count for itemset, count in itemsets.items() if count >= min_support}
        all_frequent_itemsets.update(itemsets)
        k += 1
```
This block of code filters the `itemsets` dictionary to only include itemsets with a count greater than or equal to the minimum support. It then updates the `all_frequent_itemsets` dictionary with the new frequent itemsets. Finally, it increments the size `k` by 1 for the next iteration.


```python
    return all_frequent_itemsets
```
This line returns the `all_frequent_itemsets` dictionary, which contains all frequent itemsets for the given transactions and minimum support.

---
## Function: generate_association_rules

```python
def generate_association_rules(frequent_itemsets, min_confidence):
    rules = []
    for itemset in frequent_itemsets:
        if len(itemset) > 1:
            for antecedent in combinations(itemset, len(itemset) - 1):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                confidence = frequent_itemsets[itemset] / frequent_itemsets[antecedent]
                if confidence >= min_confidence:
                    rules.append((antecedent, consequent, confidence))
    return rules
```

This function takes a dictionary of frequent itemsets and a minimum confidence threshold as input. It returns a list of association rules.


```python
def generate_association_rules(frequent_itemsets, min_confidence):
```
This line defines the function `generate_association_rules` that takes two parameters:
- `frequent_itemsets` : a dictionary of frequent itemsets
- `min_confidence` : the minimum confidence threshold for the association rules

```python
    rules = []
```
This line initializes an empty list `rules` to store the generated association rules.


```python
    for itemset in frequent_itemsets:
        if len(itemset) > 1:
            for antecedent in combinations(itemset, len(itemset) - 1):
```
This block of code iterates over each frequent itemset in the `frequent_itemsets` dictionary and checks if the itemset has more than one element.

If it does it generates all possible antecedents by taking combinations of the itemset with one less element than the itemset.


```python
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                confidence = frequent_itemsets[itemset] / frequent_itemsets[antecedent]
```
This block of code calculates the consequent of the rule by subtracting the antecedent from the itemset, and the confidence of the rule by dividing the support of the itemset by the support of the antecedent.

```math
confidence = Support(A ∪ B) / Support(A)
```
where $A$ is the antecedent and $B$ is the consequent.


```python
                if confidence >= min_confidence:
                    rules.append((antecedent, consequent, confidence))
```
This line checks if the confidence of the rule is greater than or equal to the minimum confidence threshold. If it is, the rule is added to the list of rules.


```python
    return rules
```
This line returns the list of generated association rules.

---
## Function: read_transactions_from_csv

```python
def read_transactions_from_csv(filename):
    transactions = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            transactions.append(row)
    return transactions
```
This function reads a CSV file and returns a list of transactions where each transaction is a list of items.

---
## Main Code

```python
filename = 'data.csv'
min_support = 2
min_confidence = 0.7

transactions = read_transactions_from_csv(filename)
frequent_itemsets = apriori(transactions, min_support)
print("Frequent Itemsets:")
for itemset, count in frequent_itemsets.items():
    print(f"{set(itemset)}: {count}")

association_rules = generate_association_rules(frequent_itemsets, min_confidence)
print("\nAssociation Rules:")
for antecedent, consequent, confidence in association_rules:
    print(f"{set(antecedent)} -> {set(consequent)} (Confidence: {confidence:.2f})")
```

This is the main code that reads transactions from a csv file, generates frequent itemsets, and association rules.

---
## Complete Code

```python
import csv
from itertools import combinations
from collections import defaultdict

def get_frequent_itemsets(transactions, min_support):
    itemsets = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            itemsets[frozenset([item])] += 1
    itemsets = {itemset: count for itemset, count in itemsets.items() if count >= min_support}
    return itemsets

def generate_candidates(prev_frequent_itemsets, k):
    candidates = set()
    itemsets_list = list(prev_frequent_itemsets)
    for i in range(len(itemsets_list)):
        for j in range(i + 1, len(itemsets_list)):
            candidate = itemsets_list[i] | itemsets_list[j]
            if len(candidate) == k:
                subsets = list(combinations(candidate, k - 1))
                if all(frozenset(subset) in prev_frequent_itemsets for subset in subsets):
                    candidates.add(candidate)
    return candidates

def apriori(transactions, min_support):
    transactions = list(map(set, transactions))
    itemsets = get_frequent_itemsets(transactions, min_support)
    all_frequent_itemsets = dict(itemsets)

    k = 2
    while itemsets:
        candidates = generate_candidates(itemsets, k)
        itemsets = defaultdict(int)

        for transaction in transactions:
            for candidate in candidates:
                if candidate.issubset(transaction):
                    itemsets[candidate] += 1
        itemsets = {itemset: count for itemset, count in itemsets.items() if count >= min_support}
        all_frequent_itemsets.update(itemsets)
        k += 1

    return all_frequent_itemsets

def generate_association_rules(frequent_itemsets, min_confidence):
    rules = []
    for itemset in frequent_itemsets:
        if len(itemset) > 1:
            for antecedent in combinations(itemset, len(itemset) - 1):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                confidence = frequent_itemsets[itemset] / frequent_itemsets[antecedent]
                if confidence >= min_confidence:
                    rules.append((antecedent, consequent, confidence))
    return rules

def read_transactions_from_csv(filename):
    transactions = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            transactions.append(row)
    return transactions

filename = 'data.csv'
min_support = 2
min_confidence = 0.7

transactions = read_transactions_from_csv(filename)
frequent_itemsets = apriori(transactions, min_support)
print("Frequent Itemsets:")
for itemset, count in frequent_itemsets.items():
    print(f"{set(itemset)}: {count}")

association_rules = generate_association_rules(frequent_itemsets, min_confidence)
print("\nAssociation Rules:")
for antecedent, consequent, confidence in association_rules:
    print(f"{set(antecedent)} -> {set(consequent)} (Confidence: {confidence:.2f})")
```