import pandas as pd
from scipy.stats import chi2_contingency

df = pd.read_csv('EXP_06-Correlation_coefficient_Covariance/B-Nominal/data.csv')
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