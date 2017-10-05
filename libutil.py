import csv
import numpy as np
import tabulate as tab
import matplotlib.pyplot as plt

def readCSV(path, delimitMark=',', quoteMark='|', display=False):
    data = []
    with open(path, 'rb') as csvfile:
        csvReader = csv.reader(csvfile, delimiter=delimitMark,\
                quotechar=quoteMark)
        for ele in csvReader :
            if display:
                print ('data read: {}'.format(ele))
            # concatenate
            data = data + ele
    return data

def tabularize(headers, listOfIndex, listOfData, tableFmt='orgtbl', align="right"):
    rowsOfTbl = [0] * len(listOfData)
    for rows in range(len(listOfData)) :
        rowsOfTbl[rows] = listOfData[rows]
        rowsOfTbl[rows].insert(0, listOfIndex[rows])
    tbl = tab.tabulate(rowsOfTbl, headers, tablefmt=tableFmt, numalign=align)
    return tbl

def lineChart(listOfData, listOfXaxis, listOfStyles, listOfLabels, plotTitle=None, xLabel=None, yLabel=None,\
        savePath=None, xMin=None, xMax=None, yMin=None, yMax=None, display=False):

    fig = plt.figure()
    ax = plt.gca()
    for ind, data in enumerate(listOfData) :
        plt.plot(listOfXaxis[ind], data, listOfStyles[ind], label=listOfLabels[ind])
    ax.set_xlim(xmin=xMin, xmax=xMax)
    ax.set_ylim(ymin=yMin, ymax=yMax)
    plt.title(plotTitle)
    plt.legend()
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.savefig(savePath)
    if display :
        plt.show()

    return fig

'''
Description: Serveral statistics of each column of the input data sets are
             calculated and displayed as box plot.
listOfData:  Each element contains different kinds of data.
fig:         figure object
'''

def boxPlot(listOfData, listOfXaxis=None, plotTitle=None, xLabel=None, yLabel=None,\
        savePath=None, xMin=None, xMax=None, yMin=None, yMax=None,\
        gridFlag=False, display=False):

    # multiple box plots on one figure
    fig = plt.figure()
    ax = plt.gca()
    plt.boxplot(listOfData)
    plt.xticks(range(1,len(listOfXaxis)+1), listOfXaxis)
    ax.set_xlim(xmin=xMin, xmax=xMax)
    ax.set_ylim(ymin=yMin, ymax=yMax)
    plt.title(plotTitle)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.grid(gridFlag)
    plt.savefig(savePath)
    if display :
        plt.show()

    return fig

if __name__ == "__main__" :

    '''
    Test code
    '''
    listOfData = [[24,18,19], [19,34,35]]
    listOfIndex = ['Alice', 'Bob']
    headers = ['Name', 'Age', 'dsf', 'ewo']
    tbl = tabularize(headers, listOfIndex, listOfData, 'orgtbl')
    print tbl
