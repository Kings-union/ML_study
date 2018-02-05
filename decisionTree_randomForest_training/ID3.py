#!/usr/bin/env python
# encoding:utf-8

from math import log


def calEntropy(dataSet):
    """calcuate entropy(s)
       @dateSet a training set
    """
    size = len(dataSet)
    laberCount = {}
    for item in dataSet:
        laber = item[-1]
        if laber not in laberCount.keys():
            laberCount[laber] = 0
        laberCount[laber] += 1
    entropy = 0.0
    for laber in laberCount:
        prob = float(laberCount[laber])/size
        entropy -= prob * log(prob, 2)
    return entropy


def splitDataSet(dataSet, i, value):
    """split data set by value with a laber
       @dataSet a training sets
       @i the test laber axis
       @value the test value
    """
    retDataSet = []
    for item in dataSet:
        if item[i] == value:
            newData = item[:i]
            newData.extend(item[i+1:])
            retDataSet.append(newData)
    return retDataSet


def chooseBestLaber(dataSet):
    """choose the best laber in labers
       @dataSet a traing set
    """
    numLaber = len(dataSet[0]) - 1
    baseEntropy = calEntropy(dataSet)
    maxInfoGain = 0.0
    bestLaber = -1
    size = len(dataSet)
    for i in range(numLaber):
        uniqueValues = set([item[i] for item in dataSet])
        newEntropy = 0.0
        for value in uniqueValues:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = float(len(subDataSet))/size
            newEntropy += prob * calEntropy(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > maxInfoGain:
            maxInfoGain = infoGain
            bestLaber = i
    return bestLaber


class Node:
    """the node of tree"""
    def __init__(self, laber, val):
        self.val = val
        self.left = None
        self.right = None
        self.laber = laber

    def setLeft(self, node):
        self.left = node

    def setRight(self, node):
        self.right = node


signalNode = []

def generateNode(lastNode, dataSet, labers):
    leftDataSet = filter(lambda x:x[-1]==0, dataSet)
    rightDataSet = filter(lambda x:x[-1]==1, dataSet)
    print "left:", leftDataSet
    print "right:", rightDataSet
    print "labers:", labers

    if len(leftDataSet) == 0 and len(rightDataSet) == 0:
        return

    next = 0
    print "%s ->generate left"%lastNode.laber
    if len(leftDataSet) == 0:
        print ">>> pre:%s %d stop no"%(lastNode.laber, 0)
        lastNode.setLeft(Node("no", 0))
    elif len(leftDataSet) == len(dataSet):
        print ">>> pre:%s %d stop yes"%(lastNode.laber, 0)
        lastNode.setLeft(Node("yes", 0))
    else:
        laber = chooseBestLaber(leftDataSet)
        if laber == -1:
            print ">>> can't find best one"
            laber = next
            next = (next + 1)%len(labers)
        print ">>> ",labers[laber]
        leftLabers = labers[:laber] + labers[laber+1:]
        leftDataSet = map(lambda x:x[0:laber] + x[laber+1:], leftDataSet)
        node = Node(labers[laber], 0)
        lastNode.setLeft(node)
        generateNode(node, leftDataSet, leftLabers)

    print "%s ->generate right"%lastNode.laber
    if len(rightDataSet) == 0:
        print ">>> pre:%s %d no"%(lastNode.laber, 1)
        lastNode.setRight(Node("no", 1))
    elif len(rightDataSet) == len(dataSet):
        print ">>> pre:%s %d yes"%(lastNode.laber, 1)
        lastNode.setRight(Node("yes", 1))
    else:
        laber = chooseBestLaber(rightDataSet)
        if laber == -1:
            print ">>> can't find best one"
            laber = next
            next = (next + 1)%len(labers)
        print ">>> ",labers[laber]
        rightLabers = labers[:laber] + labers[laber+1:]
        rightDataSet = map(lambda x:x[0:laber] + x[laber+1:], rightDataSet)
        node = Node(labers[laber], 0)
        lastNode.setRight(node)
        generateNode(node, rightDataSet, rightLabers)


def generateDecisionTree(dataSet, labers):
    """generate a decision tree
       @dataSet a training sets
       @labers a list of feature laber
    """
    root = None
    laber = chooseBestLaber(dataSet)
    if laber == -1:
        print "can't find a best laber in labers"
        return None
    print ">>>> ",labers[laber]
    root = Node(labers[laber], 1)
    labers = labers[:laber] + labers[laber+1:]
    dataSet = map(lambda x:x[0:laber] + x[laber+1:], dataSet)
    generateNode(root, dataSet, labers)
    return root


"""
price       size   color  result
----        ----   ----   ----
cheap       big    white  like
cheap       small  white  like
expensive   big    white  like 
expensive   small  white  like

cheap       small  black  don't like
cheap       big    black  don't like
expensive   big    black  don't like
expensive   small  black  don't like

"""
dataSet = [
    [0, 1, 1, 1],
    [0, 0, 1, 1],
    [1, 1, 1, 1],
    [1, 0, 1, 1],
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 1, 0, 0],
    [1, 0, 0, 0]]

labers = ["price", "size", "color"]


if __name__ == "__main__":
    generateDecisionTree(dataSet, labers)