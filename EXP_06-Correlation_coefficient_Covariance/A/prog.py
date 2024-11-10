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