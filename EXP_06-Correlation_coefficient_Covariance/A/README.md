Correlation Coefficient and Covariance : Numerical
==

## Introduction
This code is designed to calculate the correlation coefficient and covariance between two numeric columns in a given dataset.

## Importing Libraries
```python
import pandas as pd
import numpy as np
import statistics as st
import csv
```
These libraries are used for data manipulation, numerical computations, and statistical calculations.

## Loading the Dataset
```python
df = pd.read_csv('data.csv')
```
This line loads the dataset from a CSV file named 'data.csv' into a pandas DataFrame.

## Selecting Numeric Columns
```python
numeric_cols = df.select_dtypes(include=[np.number]).columns
```

`df.select_dtypes(include=[np.number])`: This part selects the columns from the DataFrame that contain numeric data types.

`df.columns`: This part gets the column names of the selected columns.

So, the entire line selects the numeric columns from the DataFrame and stores their names in the numeric_cols variable.

## Defining Statistical Functions

### Mean Function
The mean function calculates the average of a given dataset. It is defined as:

$$ \bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i $$

```python
def mean(x):
    return sum(x) / len(x)
```

### Variance Function
The variance function calculates the spread of a given dataset. It is defined as:

$$ \sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2 $$

```python
def variance(x):
    mean_x = mean(x)
    return sum((x - mean_x) ** 2) / len(x)
```

### Covariance Function
The covariance function calculates the linear relationship between two given datasets. It is defined as:

$$ \text{Cov}(X, Y) = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y}) $$

```python
def covariance(x, y):
    mean_x = mean(x)
    mean_y = mean(y)
    return sum((x - mean_x) * (y - mean_y)) / len(x)
```

### Correlation Coefficient Function
The correlation coefficient function calculates the strength and direction of the linear relationship between two given datasets. It is defined as:

$$ \rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} $$

```python
def correlation_coefficient(x, y):
    cov = covariance(x, y)
    var_x = variance(x)
    var_y = variance(y)
    return cov / np.sqrt(var_x * var_y)
```

## Calculating Correlation Coefficient and Covariance
```python
col1 = df[numeric_cols[0]]
col2 = df[numeric_cols[1]] 
corr_coef = correlation_coefficient(col1, col2)
cov = covariance(col1, col2)
```
This code calculates the correlation coefficient and covariance between the first two numeric columns in the dataset.

## Printing Results
```python
print()
print(f"Correlation Coefficient between {numeric_cols[0]} and {numeric_cols[1]}: {corr_coef}")
print(f"Covariance between {numeric_cols[0]} and {numeric_cols[1]}: {cov}")
print()
```
This code prints the calculated correlation coefficient and covariance.

## Interpreting Correlation Coefficient
```python
if corr_coef > 0:
    print(f"{numeric_cols[0]} and {numeric_cols[1]} are positively correlated.")
elif corr_coef < 0:
    print(f"{numeric_cols[0]} and {numeric_cols[1]} are negatively correlated.")
else:
    print(f"{numeric_cols[0]} and {numeric_cols[1]} are not correlated.")
```
This code interprets the correlation coefficient and prints whether the two columns are positively correlated, negatively correlated, or not correlated.

## Complete Code
```python
import pandas as pd
import numpy as np
import statistics as st
import csv


df = pd.read_csv('data.csv')

numeric_cols = df.select_dtypes(include=[np.number]).columns

def mean(x):
    return sum(x) / len(x)

def variance(x):
    mean_x = mean(x)
    return sum((x - mean_x) ** 2) / len(x)

def covariance(x, y):
    mean_x = mean(x)
    mean_y = mean(y)
    return sum((x - mean_x) * (y - mean_y)) / len(x)

def correlation_coefficient(x, y):
    cov = covariance(x, y)
    var_x = variance(x)
    var_y = variance(y)
    return cov / np.sqrt(var_x * var_y)

col1 = df[numeric_cols[0]]
col2 = df[numeric_cols[1]] 
corr_coef = correlation_coefficient(col1, col2)
cov = covariance(col1, col2)
print()
print(f"Correlation Coefficient between {numeric_cols[0]} and {numeric_cols[1]}: {corr_coef}")
print(f"Covariance between {numeric_cols[0]} and {numeric_cols[1]}: {cov}")
print()

if corr_coef > 0:
    print(f"{numeric_cols[0]} and {numeric_cols[1]} are positively correlated.")
elif corr_coef < 0:
    print(f"{numeric_cols[0]} and {numeric_cols[1]} are negatively correlated.")
else:
    print(f"{numeric_cols[0]} and {numeric_cols[1]} are not correlated.")

print()
```