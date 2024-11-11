Dissimilarity Matrix
=====



## Introduction
This code calculates the dissimilarity between different data points in a dataset. It supports nominal, numeric, and mixed data types.

## Importing Libraries
```python
import csv
```
The csv library is used to read the data from a csv file.

## Nominal Dissimilarity Function
```python
def nominal_dissimilarity(data, dissim):
    for i in range(len(data)):
        temp = []
        for j in range(i+1):
            if (data[i] == data[j]):
                temp.append(0)
            else:
                temp.append(1)
        dissim.append(temp)
    dissimilarity_matrix('Nominal', dissim)
```
This function calculates the dissimilarity between nominal data points. It creates a matrix where the value at each position (i, j) is 0 if the data points at positions i and j are the same, and 1 otherwise.

## Numeric Dissimilarity Function
```python
def numeric_dissimilarity(data, dissim):
    datarange = max(data) - min(data)
    for i in range(len(data)):
        temp = []
        for j in range(i+1):
            man_dist = (abs(data[i] - data[j])) / datarange
            temp.append(round(man_dist, 2))
        dissim.append(temp)
    dissimilarity_matrix('Numeric', dissim)
```
This function calculates the dissimilarity between numeric data points. It uses the Manhattan distance (also known as the L1 distance) to calculate the distance between each pair of data points, and then normalizes the distances by dividing by the range of the data.

### Manhattan Distance (L1 Distance)
The Manhattan distance, also known as the L1 distance, is a measure of the distance between two points in a multi-dimensional space.It is calculated as the sum of the absolute differences of their Cartesian coordinates.In the context of this code, it is used to calculate the dissimilarity between numeric data points.

```math
d = |x1 - x2| + |y1 - y2|
```

## Mixed Dissimilarity Function
```python
def mixed_dissimilarity(num_dissim, nom_dissim, dissim):
    for i in range(len(nom_dissim)):
        temp = []
        for j in range(i+1):
            temp.append(round((num_dissim[i][j] + nom_dissim[i][j]) / 2, 2))
        dissim.append(temp)
    dissimilarity_matrix('Mixed', dissim)
```
This function calculates the dissimilarity between mixed data points (i.e., data points that have both nominal and numeric attributes). It calculates the average of the nominal and numeric dissimilarities.

## Dissimilarity Matrix Function
```python
def dissimilarity_matrix(type, dissim):
    print()
    print(f'{type} dissimilarity matrix')
    print('-' * len(dissim))
    for i in dissim:
        print(i)
    print('-' * len(dissim))
    print()
```
This function prints the dissimilarity matrix for a given type of data.

## Main Program
```python
datfile = open("data.csv", 'r')
reader = csv.reader(datfile)

nominal_data = []
numeric_data = []

for row in reader:
    nominal_data.append(row[0])
    numeric_data.append(int(row[1]))

print(f'Nominal Data : {nominal_data}\nNumeric Data : {numeric_data}')

nom_dissim = []
num_dissim = []
mixed_dissim = []

nominal_dissimilarity(nominal_data, nom_dissim)
numeric_dissimilarity(numeric_data, num_dissim)
mixed_dissimilarity(num_dissim, nom_dissim, mixed_dissim)

datfile.close()
```
This is the main program that reads the data from a csv file, calculates the dissimilarities, and prints the results.

## Complete Code
```python
import csv

def nominal_dissimilarity(data,dissim):
    for i in range(len(data)):
        temp=[]
        for j in range(i+1):
            if (data[i]==data[j]):
                temp.append(0)
            else:
                temp.append(1)
        dissim.append(temp)
    dissimilarity_matrix('Nominal',dissim)


def numeric_dissimilarity(data,dissim):
    datarange =  max(data)-min(data)
    for i in range(len(data)):
        temp=[]
        for j in range(i+1):
            man_dist=(abs(data[i]-data[j]))/datarange
            temp.append(round(man_dist,2))
        dissim.append(temp)
    dissimilarity_matrix('Numeric',dissim)

    
def mixed_dissimilarity(num_dissim,nom_dissim,dissim):
    for i in range(len(nom_dissim)):
        temp=[]
        for j in range(i+1):
            temp.append(round((num_dissim[i][j]+nom_dissim[i][j])/2,2))
        dissim.append(temp)
    dissimilarity_matrix('Mixed',dissim)
  

def dissimilarity_matrix(type,dissim):
    print()
    print(f'{type} dissimilarity matrix')
    print('-'*len(dissim))
    for i in dissim:
        print(i)
    print('-'*len(dissim))
    print()

datfile=open("data.csv",'r')
reader = csv.reader(datfile)

nominal_data =[]
numeric_data = []

for row in reader:
    nominal_data.append(row[0])
    numeric_data.append(int(row[1]))

print(f'Nominal Data : {nominal_data}\nNumeric Data : {numeric_data}')

nom_dissim=[]
num_dissim=[]
mixed_dissim=[]

nominal_dissimilarity(nominal_data,nom_dissim)
numeric_dissimilarity(numeric_data,num_dissim)
mixed_dissimilarity(num_dissim,nom_dissim,mixed_dissim)


datfile.close() 
```