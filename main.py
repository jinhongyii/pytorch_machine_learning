import random
import os
import numpy as np


def loadDataSet(fileName):
    dataMat1 = []
    dataMat2 = []
    dataMat3 = []
    labelMat = []
   # fr = open(fileName)
    os.system("type "+fileName+" > result.txt")
    fr = open("result.txt")
    for line in fr.readlines():
        curLine = line.strip().split("\t")
        dataMat1.append(list(map(float, curLine[0:4])))
        dataMat2.append(list(map(float, curLine[4:10])))
        dataMat3.append(list(map(float, curLine[10:13])))
        labelMat.append(float(curLine[-1]))
    return dataMat1, dataMat2, dataMat3, labelMat


def standRegress(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


def leastsq(karr, start, end, xArr1, xArr2, xArr3, yArr):
    # xArr, yArr = loadDataSet(filename)
    xarr1 = xArr1[:]
    xarr2 = xArr2[:]
    xarr3 = xArr3[:]
    yarr = yArr[:]
    if start == 0:
        for (num, data) in enumerate(xarr2):
            multi1 = 0
            multi2 = 0
            for i in range(4, 10):
                multi1 += karr[i] * data[i - 4]
            for i in range(10, 13):
                multi2 += karr[i] * xarr3[num][i - 10]
            yarr[num] /= (multi2 * multi1)
        ws = standRegress(xarr1, yarr)
    elif start == 4:
        for (num, data) in enumerate(xarr1):
            multi1 = 0
            multi2 = 0
            for i in range(0, 4):
                multi1 += karr[i] * data[i]
            for i in range(10, 13):
                multi2 += karr[i] * xarr3[num][i - 10]
            yarr[num] /= (multi2 * multi1)
        ws = standRegress(xarr2, yarr)
    elif start == 10:
        for (num, data) in enumerate(xarr2):
            multi1 = 0
            multi2 = 0
            for i in range(4, 10):
                multi1 += karr[i] * data[i - 4]
            for i in range(0, 4):
                multi2 += karr[i] * xarr1[num][i]
            yarr[num] /= (multi2 * multi1)
        ws = standRegress(xarr3, yarr)

    sum = 0
    for i in range(start, end):
        sum += ws.tolist()[i-start][0] - karr[i]
    return sum, ws


xArr1, xArr2, xArr3, yArr = loadDataSet("cp.txt")

random = random.Random()
random.seed()

karr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(10):
    karr[i] = random.random()

while True:
    flag1 = False
    flag2 = False
    flag3 = False
    sum, ws = leastsq(karr, 10, 13, xArr1, xArr2, xArr3, yArr)
    if sum < 1e-6:
        flag1 = True
    for i in range(10, 13):
        karr[i] = ws.tolist()[i - 10][0]
    sum, ws = leastsq(karr, 0, 4, xArr1, xArr2, xArr3, yArr)
    if sum < 1e-6:
        flag2 = True
    for i in range(0, 4):
        karr[i] = ws.tolist()[i][0]
    sum, ws = leastsq(karr, 4, 10, xArr1, xArr2, xArr3, yArr)
    if sum < 1e-6:
        flag3 = True
    for i in range(4, 10):
        karr[i] = ws.tolist()[i - 4][0]
    if flag1 and flag2 and flag3:
        break

for i in karr:
    print(i)
