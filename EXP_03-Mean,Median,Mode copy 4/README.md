Mean, Median, Mode
================

Introduction
------------

This code is designed to calculate the mean, median, and mode of a given set of numbers. It takes the number of values and the values themselves as input from the user, sorts the data, and then calculates the mean, median, and mode.

Input Section
-------------

```python
n = int(input("Enter number of values: "))
data = []
print("Enter data : ")
for i in range(n):
    num = int(input())
    data.append(num)
```

This section of the code takes the number of values (`n`) as input from the user and then takes each value individually, storing them in a list called `data`.

Mean Calculation
----------------

```python
total = 0
for i in range(n):
    total = total + data[i]
mean = total / n
```

The mean is calculated by summing up all the values in the `data` list and then dividing by the total number of values (`n`). The formula for the mean is:

```math
μ = (Σx) / n
```

where μ is the mean, Σx is the sum of all values, and n is the total number of values.

Median Calculation
-----------------

```python
for i in range(n-1):
    for r in range(n-1-i):
        if (data[r] > data[r+1]):
            temp = data[r]
            data[r] = data[r+1]
            data[r+1] = temp
if (n % 2 == 1):
    median = data[int(n/2)]
else:
    median = ((data[int(n/2)-1] + data[int(n/2)]) / 2)
```

The median is calculated by first sorting the `data` list in ascending order. If the total number of values (`n`) is odd, the median is the middle value. If `n` is even, the median is the average of the two middle values. The formula for the median is:

$$
\begin{aligned}
M &= x({\frac{n}{2}}) \quad  \text{if } n \text{ is odd} \\
M &= \frac{x({\frac{n}{2}-1)} + x({\frac{n}{2}})}{2} \quad \text{if } n \text{ is even}
\end{aligned}
$$

where M is the median, x is the sorted list of values, and n is the total number of values.

Mode Calculation
----------------

```python
d = {}
for i in data:
    count = 0
    for j in data:
        if (i == j):
            count += 1
    d[i] = count
mode = data[0]
for i in data:
    if (d[i] > d[mode]):
        mode = i
```

The mode is calculated by creating a dictionary (`d`) where the keys are the values in the `data` list and the values are the counts of each value. The mode is the value with the highest count.

Output Section
--------------

```python
print(f"\n Sorted data : {data}")
print(f"\n Mean : {mean}\n Median : {median}\n Mode : {mode}")
```

This section of the code prints out the sorted `data` list, the mean, median, and mode.

Complete Code
--
```python
n = int(input("Enter number of values: "))
data=[]
print("Enter data : ")
for i in range(n):
	num=int(input())
	data.append(num)

#mean
total=0
for i in range(n):
	total=total+data[i]
mean = total/n	

#median
for i in range(n-1):
	for r in range(n-1-i):
		if (data[r]>data[r+1]):
			temp=data[r]
			data[r]=data[r+1]
			data[r+1]=temp
if (n % 2 == 1):
	median=data[int(n/2)]
else:
	median = (((data[int(n/2)-1]+data[int(n/2)]))/2)

#mode
d={}
for i in data:
	count=0
	for j in data:
		if(i==j):
			count+=1
	d[i]=count
mode=data[0]
for i in data:
	if(d[i]>d[mode]):
		mode=i
	
#output
print(f"\n Sorted data : {data}")
print(f"\n Mean : {mean}\n Median : {median}\n Mode : {mode}")
```