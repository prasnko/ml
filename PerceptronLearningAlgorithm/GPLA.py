#-------------------------------------------------------------------------------
# Name:        General Perceptron Learning Algorithm
# Purpose:     Machine Learning HW 3
#
# Author:      PRASANNA
#
# Created:     26/02/2013
# Copyright:   (c) PRASANNA 2013
# Licence:     PVK
#-------------------------------------------------------------------------------

#---------------Accuracy--------------------------------------------------------
# Learning Rate = 0.5 Iterations= 1000
# Training file   Testing file  Accuracy
# train.dat       train.dat     93.33%
# train.dat       test.dat      85.71%
# train2.dat      train2.dat    85.75%
# train2.dat      test2.dat     80%
# train3.dat      train3.dat    100%
# train3.dat      test3.dat     100%
# train4.dat      train4.dat    87%
# train4.dat      test4.dat     70%
#-------------------------------------------------------------------------------


import re
import math
import random
import sys

class GPLA:

 def __init__(self):
  self.weights = []

 def sigmoid(self,y):
    return float(1.0/float(1.0+float(math.exp(-y))))

 def perceptronAlgo(self,X,nInputs,Target,LearningRate,Iterations):
   output = 0.0
   Err = 0.0
   it = 1
   for i in range(nInputs+1):
##    self.weights.append(float(random.uniform(0.0,1.0)))
    self.weights.append(0.0)
   while Iterations>0:
    idx = 0
    print 'Iteration: '+ str(it)
    for i in range(len(X)):
      weightedInput = 0.0
      for j in range(nInputs):
##       weight = float(self.weights[j])
       weightedInput += self.weights[j] * float(X[i][j])
      #Adding bias weight
      weightedInput += self.weights[nInputs]
##       self.weights[j] = str(weight)
      output =  float(1.0/float(1.0+float(math.exp(-weightedInput))))

      print 'Training output: '+ str(idx+1)
      print output
      print Target[idx]
      Err = float(Target[idx]) - float(output)
      for j in range(nInputs):
##        weight = float(self.weights[j])
        self.weights[j] += float((LearningRate) * (Err )* (output) * (1.0-output) * (float(X[i][j])))
      self.weights[nInputs] += float((LearningRate) * (Err )* (output) * (1.0-output))
##        self.weights[j] = str(weight)
      idx += 1
      print  'Weights: '+ str(idx)
      print self.weights
    Iterations-=1
    it+=1

 def test(self,Y,nInputs,Target):
    print 'Testing:'
    print 'Weights:'+str(self.weights)
    outputVals = []
    for i in range(len(Y)):
      print 'Test Instance:'+str(i+1)
      weightedInput = 0.0
      for j in range(nInputs):
       weight = float(self.weights[j])
       weightedInput += weight * float(Y[i][j])
      #Adding bias weight
      weightedInput+= float(self.weights[nInputs])
      print 'Weighted Input:' +str(weightedInput)
      output = float(1.0/float(1.0+float(math.exp(-weightedInput))))
      print 'Output:' +str(output)
      if output>= 0.5:
       output = 1
      else:
       output = 0
      outputVals.append(output)
    correct = 0
    for i in range(len(Target)):
        if str(outputVals[i]) == str(Target[i]):
            correct +=1
    accuracy = 0
    accuracy = float(float(correct)/float(len(Target))) * 100
    print 'Accuracy: ' + str(accuracy) +' %'

 def formatFile(self,f,train):
  X = []
  trainTarget = []
  testTarget = []
  attr_name = []
  attributes = []
  text = []
  words = []
  text = f.read()
  lines = text.split('\n')
    # spliting text into a list of words
  attr_name = re.split('\W+',lines[0])
    # Extracting the attribute names
  for attr in attr_name:
   attributes.append(attr)
  lines.remove(str(lines[0]))
  lines.remove('')
  for line in lines:
    if(str(line)!=''):
     words.append(line.split('\t'))
  for word in words:
   if str(word[:-1]) !='':
    X.append(word[:-1])
   if train == 0:
    if str(word[-1]) != '':
     testTarget.append(word[-1])
   else:
    if str(word[-1]) != '':
     trainTarget.append(word[-1])
  if train:
    return X, trainTarget, len(X[1])
  else:
    return X, testTarget, len(X[1])

def main():
 X = []
 Y = []
 trainTarget = []
 testTarget = []
 train = open(sys.argv[1])
 test = open(sys.argv[2])
 learningRate = float(sys.argv[3])
 iterations = int(sys.argv[4])
## train = open(r'C:\Prasanna\Spring13\ML\HW3\data\train4.dat')
## test = open(r'C:\Prasanna\Spring13\ML\HW3\data\train4.dat')
## learningRate = 0.5
## iterations = 1000
 g = GPLA()
 X, trainTarget, nInputsTrain = g.formatFile(train,1)
 Y,testTarget, nInputsTest = g.formatFile(test,0)
## print Y
## print testTarget
## print Target
## for x in X:
##    print x
#### print Target
 g.perceptronAlgo(X,nInputsTrain,trainTarget,learningRate,iterations)
 g.test(Y,nInputsTest,testTarget)

if __name__ == '__main__':
    main()
