Statistical Description
================

Table of Contents
-----------------

- [Statistical Description](#statistical-description)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Importing Libraries](#importing-libraries)
  - [Loading Data](#loading-data)
  - [Calculating Quantiles](#calculating-quantiles)
  - [Removing Outliers](#removing-outliers)
  - [Calculating Mean](#calculating-mean)
  - [Calculating Median](#calculating-median)
  - [Calculating Mode](#calculating-mode)
  - [Calculating Standard Deviation](#calculating-standard-deviation)
  - [Complete Code](#complete-code)

Introduction
------------

This code is designed to load a dataset from a CSV file, calculate various statistical measures, and remove outliers from the data.

Importing Libraries
-------------------

```python
import numpy as np
import math
from collections import Counter
```

This section imports the necessary libraries for the code to function. `numpy` is used for numerical computations, `math` is used for mathematical functions, and `Counter` is used to count the frequency of elements in a list.

Loading Data
-------------

```python
datfile = open("data.csv", 'r')
datastr = datfile.read()
datastr = datastr.strip()
datalist = datastr.split(',')
data = [int(i) for i in datalist]
```

This section loads the data from a CSV file named "data.csv". The data is read as a string, stripped of any leading or trailing whitespace, split into a list of strings using the comma as a delimiter, and then converted to a list of integers.

Calculating Quantiles
---------------------

```python
lowerq = np.quantile(data, 0.25)
upperq = np.quantile(data, 0.75)
iqr = upperq - lowerq
loweroutlier = lowerq - 1.5 * iqr
upperoutlier = upperq + 1.5 * iqr
```

This section calculates the lower and upper quartiles of the data using the `np.quantile` function. The interquartile range (IQR) is then calculated as the difference between the upper and lower quartiles. The lower and upper outlier thresholds are calculated as 1.5 times the IQR below and above the lower and upper quartiles, respectively.

Removing Outliers
-----------------

```python
outliers = []
newdata = []
for i in data:
    if (i > loweroutlier and i < upperoutlier):
        newdata.append(i)
    else:
        outliers.append(i)
```

This section removes outliers from the data by iterating over each element in the data list. If the element is within the range defined by the lower and upper outlier thresholds, it is added to the `newdata` list. Otherwise, it is added to the `outliers` list.

Calculating Mean
----------------

```python
mean = np.mean(data)
```

This section calculates the mean of the data using the `np.mean` function.

Calculating Median
-----------------

```python
data.sort()
median = np.median(data)
```

This section sorts the data in ascending order and then calculates the median using the `np.median` function.

Calculating Mode
----------------

```python
frequency = Counter(data)
max_freq = max(frequency.values())
modes = [k for k, v in frequency.items() if v == max_freq]
num_modes = len(modes)
if num_modes == 1:
    mode_type = "unimodal"
elif num_modes == 2:
    mode_type = "bimodal"
elif num_modes == 3:
    mode_type = "trimodal"
else:
    mode_type = "multimodal"
```

This section calculates the frequency of each element in the data using the `Counter` class. The maximum frequency is then determined, and the modes are identified as the elements with the maximum frequency. The number of modes is then determined, and the mode type is classified as unimodal, bimodal, trimodal, or multimodal based on the number of modes.

Calculating Standard Deviation
------------------------------

```python
std_dev = np.std(data)
```

This section calculates the standard deviation of the data using the `np.std` function.

Complete Code
------------------------

```python
import numpy as np
import math
from collections import Counter

datfile=open("data.csv",'r')
datastr=datfile.read() 
datastr=datastr.strip()
datalist=datastr.split(',')
data =[int(i) for i in datalist]

#quantiles
lowerq=np.quantile(data,0.25)
upperq=np.quantile(data,0.75)
iqr= upperq-lowerq
loweroutlier = lowerq-1.5*iqr
upperoutlier = upperq+1.5*iqr

print(f"Lower Quantile = {lowerq}\nUpper Quantile = {upperq}\nInter  Quantile Range = {iqr}\nLower Outlier = {loweroutlier}\nUpper Outlier = {upperoutlier}\n")

outliers = []
newdata = []
for i in data:
	if (i>loweroutlier and i < upperoutlier):
		newdata.append(i)
	else: 
		outliers.append(i)	
	
print(f"Outliers = {outliers}\n\nData without Outliers = {newdata}\n\nMaximum value = {max(newdata)}\nMinimum value = {min(newdata)}\n")

data=newdata

#mean
mean = np.mean(data)
print(f"Mean = {mean}")

#median
data.sort()
median =np.median(data)
print(f"Median = {median}")

#mode
frequency = Counter(data)
max_freq = max(frequency.values())
modes = [k for k, v in frequency.items() if v == max_freq]
num_modes = len(modes)
if num_modes == 1:
    mode_type = "unimodal"
elif num_modes == 2:
    mode_type = "bimodal"
elif num_modes == 3:
    mode_type = "trimodal"
else:
    mode_type = "multimodal"  
print(f"Mode(s): {modes} {max_freq} times")
print(f"Mode Type: {mode_type}")


#standard deviation
std_dev=np.std(data)
print(f"Standard deviation: {std_dev}")

```