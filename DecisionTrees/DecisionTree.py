#-------------------------------------------------------------------------------
# Name:        Decision Trees
# Purpose:     Machine Learning Homework 1
#
# Author:      PRASANNA
#
# Created:     04/02/2013
# Copyright:   (c) PRASANNA 2013
# Licence:     PVK
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
# B> Accuracy of decision tree on training file train.txt = 87.5%
#
# C> Accuracy of decision tree on test file test.txt = 83.25%
#
# Data Structures and Functions
#
# calcEntropy -> Calculates the entropy for the entire dataset that is passes to it based on the class values and returns it.
# labelCounts stores the counts of each label
#
# splitDataset -> Splits the dataset based on the attribte index and it's value both of which are passed to it and returns the sub dataset.
# newFeatVec stores a list of features without the attribute value that is matched.
# newX stores the reduced dataset.
#
# getBestFeature -> Returns the feature that provides the maximum information gain.
# featList stores the feature list for each attribute.
# subDataset stores the reduced dataset that is returned by SplitDataset function
#
# majorityCount-> Returns the class label that has majority of the examples.
# classCount dictionary is used to store the count of examples of each class value with class value as key and count as value.
# checkCompList is passed to check if we should check count of complete list in case of tie among subset of class values for examples at leaf.
#
# createTree -> It is used to create the decision tree. It return the decision tree that is created in the form of a dictionary.
# decisionTree is a dictionary that is used to build the decision Tree recursively by storing another dictionary for next level.
# labels stores a list of attribute names.
# level variable helps in formating the decision tree based on the level of the recursive call
#
# classify -> It is used to classify a given feature vector and returns the class label that the feature vector belongs to.
# classList is a list that stores class values of entire dataset
#
# calcAccuracy -> Calculates decision tree accuracy using actual and predicted class labels that are passed in the form of lists
# and returns count of correctly classified instances.
#
# formatText -> Formats the text document that is passed to it to return two lists:
# words: which is training data in the form of list of lists.
# attributes: which is a list of names of the features.
#
# partialSetTest -> part d is carried out with the help of this method. It generates the training data and attributes and then generates
# decision tree and calculates accuracy on increasing subsets of dataset by size 50. Finally, it plots the result using matplotlib library.
#
#-------------------------------------------------------------------------------


from math import log
import operator
import re
import sys
from sys import stdout
import numpy
import matplotlib.pyplot as plt
import pylab as pl

def calcEntropy(dataSet):
 numEntries = len(dataSet)
 labelCounts = {}
 # Classifying each feature vector in the dataset based on the class label
 for featVec in dataSet:
  currentLabel = featVec[-1]
  if currentLabel not in labelCounts.keys():
   labelCounts[currentLabel] = 0
  labelCounts[currentLabel] += 1
 entropy = 0.0
 # Calculating probability and then the entropy
 for key in labelCounts:
  prob = float(labelCounts[key])/numEntries
  entropy -= prob * log(prob,2)
 return entropy

def splitDataSet(X, attrIndex, value):
 newX = []
 for featVec in X:
  # Keeping only those feature vectors whose feature value matches the required but removing the value from the vector.
  if featVec[attrIndex] == value:
   newFeatVec = featVec[:attrIndex]
   newFeatVec.extend(featVec[attrIndex+1:])
   newX.append(newFeatVec)
 return newX

def getBestFeature(dataSet):
 numFeatures = len(dataSet[0]) - 1
 entropy = calcEntropy(dataSet)
 bestInfoGain = 0.0; bestFeature = -1
 # Check each feature in the dataset, if it's information gain is highest.
 for i in range(numFeatures):
  featList = [example[i] for example in dataSet]
  uniqueVals = set(featList)
  newEntropy = 0.0
 # For each feature value, split the dataset on that value and calculate the entropy on it.
  for value in uniqueVals:
   subDataSet = splitDataSet(dataSet, i, value)
   prob = len(subDataSet)/float(len(dataSet))
   newEntropy += prob * calcEntropy(subDataSet)
  infoGain = entropy - newEntropy
 # Highest information gain providing feature is the best attribute.
  if (infoGain > bestInfoGain):
    bestInfoGain = infoGain
    bestFeature = i
 return bestFeature

def majorityCount(classList,checkCompList):
 classCount={}
 for vote in classList:
  if vote not in classCount.keys():
   classCount[vote] = 0
  classCount[vote] += 1
  # Sorting ensures we have maximum value at index 0
 sortedClassCount = sorted(classCount.iteritems(),
  key=operator.itemgetter(1), reverse=True)
 if len(sortedClassCount)>1:
 # If both classes are equally distributed then return -1 so maximum among entire dataset can be taken.
  if sortedClassCount[0][1]==sortedClassCount[1][1] and checkCompList==0:
    return -1
  else:
    return sortedClassCount[0][0]
 else:
    return sortedClassCount[0][0]


def createTree(dataSet,labels,wholeClassList,wholeDataSet,level):
## if len(dataSet)> len(wholeDataSet):
##  wholeDataSet = dataSet[:]
 level+=1
 classList = [example[-1] for example in dataSet]
 if len(classList) > len(wholeClassList):
    wholeClassList = classList[:]
 if len(classList)>0:
  if classList.count(classList[0]) == len(classList):
   return classList[0]
 if len(dataSet)>0:
 # If no more examples remain, take maximum of class values at leaf node else maximum from whole dataset.
  if len(dataSet[0]) == 1:
   leafCount = majorityCount(classList,0)
   if leafCount !=-1:
    return leafCount
   else:
    return majorityCount(wholeClassList,1)
 else:
    return majorityCount(wholeClassList,1)
 bestFeat = getBestFeature(dataSet)
 bestFeatLabel = labels[bestFeat]
 # Store decision tree recursively in dictionary.
 decisionTree = {bestFeatLabel:{}}
 del(labels[bestFeat])
 featValues = [example[bestFeat] for example in wholeDataSet]
 # Extracting unique value of features for given attribute.
 uniqueVals = set(featValues)
 # For each value of the best feature selected, generate the tree.
 for value in uniqueVals:
  treeOutput = ''
  if level==1:
    stdout.write('\n')
  for i in range(0,level-1):
    if i==0 :
     stdout.write("\n")
    treeOutput += '| '
  treeOutput += bestFeatLabel+'='+str(value)
  stdout.write("%s" %treeOutput)
  subLabels = labels[:]
  decisionTree[bestFeatLabel][value] = createTree(splitDataSet\
(dataSet, bestFeat, value),subLabels,wholeClassList,wholeDataSet,level)
 # If value returned from lower level is a number return it as class label.
  if type(decisionTree[bestFeatLabel][value]).__name__ != 'dict':
     stdout.write(":%d" % int(decisionTree[bestFeatLabel][value]))
 return decisionTree

# Classifying one feature vector at a time
def classify(inputTree,attributes,featVec,wholeClassList):
 firstLevel = inputTree.keys()[0]
 secondLevel = inputTree[firstLevel]
 # Feature index of attribute selected at first level
 featIndex = attributes.index(firstLevel)
 # Traversing down the tree recursively
 k= secondLevel.keys()
 i=1
 clAssigned =0
 # For keys at the next level if a match is found process further
 for key in secondLevel.keys():
  if featVec[featIndex] == key:
    clAssigned = 1
 # If key type is dictionary it means next level is not a leaf node, so process recursively.
    if type(secondLevel[key]).__name__=='dict':
     classLabel = classify(secondLevel[key],attributes,featVec,wholeClassList)
    else:
 # At leaf level assign the class label
     classLabel = secondLevel[key]
  elif i == len(k) and clAssigned == 0:
   classLabel = majorityCount(wholeClassList,1)
  i+=1
 return classLabel

def calcAccuracy(predicted,actual):
    accCount=0
    for i in range(len(actual)):
        if predicted[i] == actual[i]:
         accCount=accCount+1
    return accCount


def formatText(x):
    lines = []
    attributes = []
    words = []
    text = x.read()
    lines = text.split('\n')
    # spliting text into a list of words
    attr_name = re.split('\W+',lines[0])
    i=0
    # Extracting the attribute names
    for attr in attr_name:
      if i%2==0:
       attributes.append(attr)
      i= i+1
    # Removing the attribute name from training data.
    lines.remove(str(lines[0]))
    lines.remove('')
    # Storing each feature vector in a list.
    for line in lines:
     words.append(line.split('\t'))
    return words,attributes

# Function to check effect of training data size on accuracy and drawing learning curve.
def partialSetTest():
    lines = []
    attributes = []
    words = []
    line =[]
    actual = []
    predicted = []
    leafValues = []
    wholeClassList = []
    wholeDataset = []
    X=[]
    Y=[]
##    train = open(r'C:\Prasanna\Spring13\ML\HW1\data\train.txt')
##    test = open(r'C:\Prasanna\Spring13\ML\HW1\data\test.txt')
    train = open(sys.argv[1])
    test = open(sys.argv[2])
    wordsTrain,attributesTrain = formatText(train)
    wordsTest,attributesTest = formatText(test)
    for j in range(50,len(wordsTrain),50):
     X.append(j)
     partialTrainSet = wordsTrain[0:j]
     attr = attributesTrain[:]
     tree = createTree(partialTrainSet,attr,wholeClassList,partialTrainSet,0)
##    leafValues,wholeClassList
 # Actual class label
     for word in wordsTest:
      actual.append(word[-1])
     featLabels = attributesTest[:]
     wholeClassList = [example[-1] for example in partialTrainSet]
     for word in wordsTest:
      featVec = word[:-1]
      predicted.append(classify(tree,featLabels,featVec,wholeClassList))
     accCount = calcAccuracy(predicted,actual)
     Y.append(float(accCount)*100/float(len(actual)))
    pl.xlabel('Training set size')
    pl.ylabel('Accuracy')
    pl.title('Learning Curve')
##    pl.xlim(0,900)
##    pl.ylim(0,100)
   # Matplotlib method to plot and show the data.
    pl.plot(X,Y)
    pl.show()


def main():
##    partialSetTest()
    lines = []
    attributes = []
    words = []
    line =[]
    actual = []
    predicted = []
    leafValues = []
    wholeClassList = []
    wholeDataset = []
##    train = open(r'C:\Prasanna\Spring13\ML\HW1\data\train.txt')
##    test = open(r'C:\Prasanna\Spring13\ML\HW1\data\test.txt')
    train = open(sys.argv[1])
    test = open(sys.argv[2])
    # Formating the train and text documents in the form of lists for processing in the algorithm
    wordsTrain,attributesTrain = formatText(train)
    wordsTest,attributesTest = formatText(test)
    print 'Training Instances size:'+ str(len(wordsTrain))
    print 'Attributes:'+str(len(attributesTrain))
    for attr in attributesTrain:
        print attr
    print 'Testing Instances size:'+ str(len(wordsTest))
    # Sending second copy of training data as feature values need to be iterated on whole training dataset and not the reduced dataset which would miss out some values in the decision tree.
    tree = createTree(wordsTrain,attributesTrain,wholeClassList,wordsTrain,0)
    # Actual class label
    for word in wordsTest:
        actual.append(word[-1])
    featLabels = attributesTest[:]
    wholeClassList = [example[-1] for example in wordsTrain]
    # Generating list of predicted class values using classify function.
    for word in wordsTest:
     featVec = word[:-1]
     predicted.append(classify(tree,featLabels,featVec,wholeClassList))
    accCount = calcAccuracy(predicted,actual)
    print '\n'+'Correctly Classified Instances: '+str(accCount)
    print 'Incorrectly Classified Instances: '+str(len(actual)-accCount)
    print 'Accuracy= '+ str(float(accCount)*100/float(len(actual))) +' %'



if __name__ == '__main__':
    main()
