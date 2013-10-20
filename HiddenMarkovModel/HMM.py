#-------------------------------------------------------------------------------
# Name:        Viterbi Decoding Algorithm
# Purpose:     Machine Learning course
#
# Author:      PRASANNA
#
# Created:     13/03/2013
# Copyright:   (c) PRASANNA 2013
# Licence:     prsko
#-------------------------------------------------------------------------------
import re
import sys

#-------------------------------Explanation-------------------------------------
#
#   Data Structures:
# self.viterbi :  dictionary used to store viterbi value with key-state,time & value- viterbi value computed at that time in that state
# self.backpointer: dictionary used to store the state having maximum viterbi probability at previous time
# self.a: dictionary used to store transition probabilities from state s-1 to s(key- s-1,s)
# self.b: dictionary used to store emmision probabilities with observation obs[t] at time t and state j with key-
# self.q: state sequence with key-state,time and value as state value
# self.o: observation sequence
# self.vList: used to find maximum viterbi value from previous states
#
#   Functions:
# init : data structure initialization
# formatdata: format input data and fill up the data structures
# getBestPath: used to print best path sequence from given observation sequence
#
#
#-------------------------------------------------------------------------------

class ViterbiAlgo:

   def __init__(self):
    self.viterbi = {}
    self.backpointer={}
    self.a={}           ## transition probabilities
    self.b={}           ## emssion probabilities
    self.q={}
    self.o=[]           ## observation sequence
    self.vList = {}     ## used to find maximum of previous two states

   def formatData(self,data,testData):
    lines =  data.split('\n')
    testsequences = testData.split('\n')
    testsequences.remove('')
    no=1
    for testsequence in testsequences:
     print 'Sequence '+str(no)+' '+testsequence
     no+=1
     testSequence= re.split('\W+',testsequence)
     N = int(lines[0])
     words = []
     words = lines[1].split(' ')
     i=0
     for s in range(1,N+1):
      self.a[0,s] = words[i]
      i+=1
     words = []
     words = lines[2].split(' ')
     i=0
     for sd in range(1,N+1):
      for s in range(1,N+1):
        self.a[sd,s] = words[i]
        i+=1
     nObs = int(lines[3])
     words = []
     words = lines[4].split(' ')
     obs = []
     obs.append('')
     for i in range(0,nObs):
      obs.append(words[i])
     words = []
     words = lines[5].split(' ')
     k=0
     for t in range(1,nObs+1):
      for j in range(1,N+1):
        self.b[obs[t],j] = words[k]
        k+=1
     self.o.append('')
     M = len(testSequence)
##     print M
     for t in range(1,M+1):
      for j in range(1,N+1):
            self.q[j,t] = j
     for i in range(M):
      self.o.append(testSequence[i])
##     print self.o
     self.getBestPath(N,M)



   def getBestPath(self,N,T):
        vit= {}
        ## viterbi and backpointer for start state
        for s in range(1,N+1):
            self.viterbi[s,1]=float(self.a[0,s])*float(self.b[self.o[1],s])
            self.backpointer[s,0]=''
            vit[s] = self.a[0,s]
        maxVal = max(vit.values())
        for key,value in vit.iteritems():
         if value == maxVal:
          maxState = int(key)
        for s in range(1,N+1):
         self.backpointer[s,1] = maxState
        ## viterbi and backpointer for other states
        for t in range(2,T+1):
            for s in range(1,N+1):
             for sd in range(1,N+1):
              self.vList[sd]= (float(self.viterbi[sd,t-1])*float(self.a[sd,s])*float(self.b[self.o[t],s]))   ##storing state having maximum viterbi path probability as value for viterbi path probability as key
             self.viterbi[s,t] =max(self.vList.values())                           ## getting previous state with maximum viterbi value
##             print 'Viterbi prob for s= '+ str(s) +' t= '+str(t)+' is '+str(self.viterbi[s,t])
             for state, viterbi in self.vList.iteritems():
              if viterbi == self.viterbi[s,t]:
                 smax=state
##             print 'Smax = '+str(smax)
             self.vList.clear()
             self.backpointer[s,t]=''
             self.backpointer[s,t]+= str(self.q[smax,t-1])                              ## storing the backpointer
##             print 'Backpointer for s= '+ str(s) +' t= '+ str(t)+' is '+str(self.backpointer[s,t])
        totalBackPointer = ''
##        print ''
##        print self.backpointer
##        print ''
        ## traversing from final to start state to find maximum probability path
        for t in range(T,0,-1):
            if t==T:                                                                    ## for final state(time slot)
                sb=int(self.backpointer[smax,t])
                totalBackPointer=str(smax)
##                totalBackPointer+=self.backpointer[smax,t]
##                print 'Total backpointer at t= '+str(t)+' is '+totalBackPointer
            else:                                                                       ## for other states (time slot)
                sb=int(self.backpointer[sb,t])
                totalBackPointer+=str(sb)
##                print 'Total backpointer at t= '+str(t)+' is '+totalBackPointer
##        totalBackPointer+='0'
        s = totalBackPointer
        ## reversing string to get original order of states
        s = s[::-1]
        output=''
        for c in s:
            if int(c)==1:
                output+='s1->'
            elif int(c)==2:
                output+='s2->'
            else:
                output+='s3->'
        output=output[:-2]
        print 'State sequence: ' +output



def main():
    v = ViterbiAlgo()
    model = open(sys.argv[1])
    data = model.read()
    test = open(sys.argv[2])
    testData = test.read()
    v.formatData(data,testData)
##    print 'Sequence 1 (312312312): '
##    v.sequence1()
##    print 'Sequence 2 (311233112): '
##    v.sequence2()


if __name__ == '__main__':
    main()
