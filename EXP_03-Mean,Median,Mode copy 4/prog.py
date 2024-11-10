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

