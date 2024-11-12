import numpy as np
import math
from collections import Counter

datfile=open("EXP_04-Statistical_Description/data.csv",'r')
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
