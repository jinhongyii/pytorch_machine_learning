import os
def loadDataSet(fileName):
    dataMat1 = []
    labelMat = []
    os.system("type " + fileName + " > result.txt")
    fr = open("result.txt")
    for line in fr.readlines():
        curLine = line.strip().split("\t")
        dataMat1.append(list(map(float, curLine[:-1])))
        labelMat.append([float(curLine[-1])])
    return dataMat1, labelMat

xarr,yarr=loadDataSet("test.txt")
