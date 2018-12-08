import os
import random


def loadDataSet(fileName):
    dataMat1 = []
    dataMat2 = []
    dataMat3 = []
    labelMat = []
    # fr = open(fileName)
    os.system("type " + fileName + " > result.txt")
    fr = open("result.txt")
    for line in fr.readlines():
        curLine = line.strip().split("\t")
        dataMat1.append(list(map(float, curLine[0:4])))
        dataMat2.append(list(map(float, curLine[4:10])))
        dataMat3.append(list(map(float, curLine[10:13])))
        labelMat.append(float(curLine[-1]))
    return dataMat1, dataMat2, dataMat3, labelMat


xArr1, xArr2, xArr3, yArr = loadDataSet("cp.txt")

random = random.Random()
random.seed()
alpha = 1e-4
karr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(13):
    karr[i] = random.random()
sigma = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
maxiter = 2e8
iter = 0
count=0
while True:
    flag = True
    for (num, datay) in enumerate(yArr):
        iter += 1
        count+=1
        A = karr[0]*xArr1[num][0]+karr[1]*xArr1[num][1]+karr[2]*xArr1[num][2]+karr[3]*xArr1[num][3]
        B = karr[4]*xArr2[num][0]+karr[5]*xArr2[num][1]+karr[6]*xArr2[num][2]+karr[7]*xArr2[num][3]+karr[8]*xArr2[num][4]+karr[9]*xArr2[num][5]
        C = karr[10]*xArr3[num][0]+karr[11]*xArr3[num][1]+karr[12]*xArr3[num][2]
        for i in range(0, 4):
            sigma[i] += (A * B * C - datay) * (karr[i] * B * C)
        for i in range(4, 10):
            sigma[i] += (A * B * C - datay) * (karr[i] * A * C)
        for i in range(10, 13):
            sigma[i] += (A * B * C - datay) * (karr[i] * B * A)

        if(count==46):
            count=0
            for i in range(0,13):
                sigma[i]/=46
        else:
            continue
        for i in range(0, 13):
            karr[i] = karr[i] - alpha * sigma[i]
        sum=0
        for i in sigma:
            sum+=abs(i)
        if sum<0.5:
            flag=False
            break
        #print(0.5*(A*B*C-datay)*(A*B*C-datay))
        for i in sigma:
            print(i,end=" ")
        print()
        for i in range(0,13):
            sigma[i]=0
    if (not flag) or iter > maxiter:
        break



for i in karr:
    print(i)
for i in sigma:
    print(i, end=" ")
input()
