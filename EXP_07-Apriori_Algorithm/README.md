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

## Function 1: get_frequent_itemsets

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

## Function 2: generate_candidates

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

## Function 3: apriori

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

## Function 4: generate_association_rules

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

## Function 5: read_transactions_from_csv

```python
def read_transactions_from_csv(filename):
    transactions = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            transactions.append(row)
    return transactions
```

This function takes a filename as input and returns a list of transactions read from the csv file.

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