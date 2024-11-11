Correlation Coefficient and Covariance : Nominal
==

## Introduction
This code is designed to calculate the correlation coefficient and covariance between two nominal variables using the chi-squared test. It loads a CSV file, creates a contingency table, performs the chi-squared test, and interprets the results.



## Importing Libraries
```python
import pandas as pd
from scipy.stats import chi2_contingency
```
This section imports the necessary libraries for the code. `pandas` is used for data manipulation and analysis, while `scipy.stats` is used for statistical calculations.

## Loading Data
```python
df = pd.read_csv('data.csv')
print(df)
```
This section loads a CSV file named `data.csv` into a pandas DataFrame called `df`. The `print(df)` statement is used to display the contents of the DataFrame.

## Creating a Contingency Table
```python
table_contingency = pd.crosstab(df['Val1'],df['Val2'])
```
This section creates a contingency table using the `pd.crosstab` function. The table is created based on the values in the `Val1` and `Val2` columns of the DataFrame.

#### Contigency Table
A contingency table is a type of table used in statistics to display the relationship between two categorical variables. It is also known as a cross-tabulation or crosstab. The table consists of rows and columns, where each row represents a category of one variable and each column represents a category of the other variable. The cells in the table contain the frequency or count of observations that fall into each combination of categories.

#### Example :

|       | blue | green | red  | 
|-------|------|-------|------|
| red   | 1    | 0     | 1    |
| yellow| 0    | 1     | 0    |

## Performing Chi-Squared Test
```python
stat,pval,dof,ex_freq = chi2_contingency(table_contingency)
```
This section performs a chi-squared test on the contingency table using the `chi2_contingency` function from `scipy.stats`. The function returns the chi-squared statistic, p-value, degrees of freedom, and expected frequencies.

### The Chi-Squared Test 

The Chi-Squared Test is a statistical method used to determine whether there is a significant association between two categorical variables.

```math
χ² = ∑ [(Oᵢ - Eᵢ)² / Eᵢ]
```
where Oᵢ is the observed frequency and Eᵢ is the expected frequency.

- χ² is the chi-squared statistic
- observed frequency is the actual number of observations in each category
- expected frequency is the number of observations that would be expected in each category if there were no association between the variables

The chi-squared test is commonly used to determine whether there is a significant association between two categorical variables, such as gender and voting behavior, or whether there is a significant difference between the observed frequencies and the expected frequencies in a contingency table.

## Displaying Results
```python
print(f'Contingency Table:\n {table_contingency}\n')
print(f'Chi_squared Statistic: {stat}')
print(f'p_value: {pval}')
print(f'Degrees of Freedom: {dof}')
```
This section displays the results of the chi-squared test, including the contingency table, chi-squared statistic, p-value, and degrees of freedom.

## Interpreting Results
```python
if pval< 0.5:
    print(f'There is a correlation between val1 and val2')
else:
    print(f'There is no correlation between val1 and val2')
```
This section interprets the results of the chi-squared test. If the p-value is less than 0.5, it indicates a significant correlation between the variables. Otherwise, it indicates no significant correlation.

Complete Code
--
```python
import pandas as pd
from scipy.stats import chi2_contingency

df = pd.read_csv('data.csv')
print(df)


table_contingency = pd.crosstab(df['Val1'],df['Val2'])

stat,pval,dof,ex_freq = chi2_contingency(table_contingency)

print(f'Contingency Table:\n {table_contingency}\n')

print(f'Chi_squared Statistic: {stat}')
print(f'p_value: {pval}')
print(f'Degrees of Freedom: {dof}')
if pval< 0.5:
    print(f'There is a correlation between val1 and val2')
else:
    print(f'There is no correlation between val1 and val2')
```