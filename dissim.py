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
            temp.append((man_dist//0.001)/1000)
        dissim.append(temp)
    dissimilarity_matrix('Numeric',dissim)

def mixed_dissimilarity(num_dissim,nom_dissim,dissim):
    for i in range(len(nom_dissim)):
        temp=[]
        for j in range(i+1):
            temp.append((((num_dissim[i][j]+nom_dissim[i][j])/2)//0.001)/1000)
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

datfile=open("dissim.csv",'r')
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