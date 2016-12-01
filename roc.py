import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand


# Generate ROC 
def generate_roc(scoreMatrix,trueLabels,nROCpts =100 ,plotROC = 'false'):

    tpr = np.zeros([1,nROCpts]) 
    fpr = np.zeros([1,nROCpts]) 
    nTrueLabels = np.count_nonzero(trueLabels) 
    nFalseLabels = np.size(trueLabels) - nTrueLabels 
    
    minScore = np.min(scoreMatrix)
    maxScore = np.max(scoreMatrix);
    rangeScore = maxScore - minScore;
  
    
    thdArr = minScore + rangeScore*np.arange(0,1,float(1)/(nROCpts))
    #thdArr=np.array(thdArr)
    #print thdArr
    for thd_i in range(0,nROCpts):
        thd = thdArr[thd_i]
        ind = np.where(scoreMatrix>=thd) 
        thisLabel = np.zeros([np.size(scoreMatrix,0),np.size(scoreMatrix,1)])
        thisLabel[ind] = 1
        tpr_mat = np.multiply(thisLabel,trueLabels)
        tpr[0,thd_i] = np.sum(tpr_mat)/nTrueLabels
        fpr_mat = np.multiply(thisLabel, 1-trueLabels)
        fpr[0,thd_i] = np.sum(fpr_mat)/nFalseLabels
        
        #print fpr
       # print tpr  
    if(plotROC == 'true'):
        plt.xlim(min(fpr[0,:]), max(fpr[0,:]))
        plt.xlabel('False positive Rate', fontsize=18)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.plot(fpr[0,:],tpr[0,:], 'b.-')
        
        plt.show()

    return fpr,tpr,thdArr


# def generate_roc(scoreMatrix,scoreMatrix2,trueLabels,nROCpts =100 ,plotROC = 'false'):

#     tpr = np.zeros([1,nROCpts]) 
#     fpr = np.zeros([1,nROCpts]) 
#     tpr2 = np.zeros([1,nROCpts]) 
#     fpr2 = np.zeros([1,nROCpts]) 
    
#     nTrueLabels = np.count_nonzero(trueLabels) 
#     nFalseLabels = np.size(trueLabels) - nTrueLabels 
    
#     minScore = np.min(scoreMatrix)
#     maxScore = np.max(scoreMatrix);
#     rangeScore = maxScore - minScore;
  
#   ###added
#     minScore2 = np.min(scoreMatrix2)
#     maxScore2 = np.max(scoreMatrix2);
#     rangeScore2 = maxScore2 - minScore2;
  



#     thdArr = minScore + rangeScore*np.arange(0,1,float(1)/(nROCpts))
#     thdArr2 = minScore2 + rangeScore2*np.arange(0,1,float(1)/(nROCpts))
    

#     #thdArr=np.array(thdArr)
    
#     #print thdArr
#     for thd_i in range(0,nROCpts):
#         thd = thdArr[thd_i]
#         ind = np.where(scoreMatrix>=thd) 
#         thisLabel = np.zeros([np.size(scoreMatrix,0),np.size(scoreMatrix,1)])
#         thisLabel[ind] = 1
#         tpr_mat = np.multiply(thisLabel,trueLabels)
#         tpr[0,thd_i] = np.sum(tpr_mat)/nTrueLabels
#         fpr_mat = np.multiply(thisLabel, 1-trueLabels)
#         fpr[0,thd_i] = np.sum(fpr_mat)/nFalseLabels


#         ### added
#         thd2 = thdArr2[thd_i]
#         ind2 = np.where(scoreMatrix2>=thd2) 
#         thisLabel2 = np.zeros([np.size(scoreMatrix2,0),np.size(scoreMatrix2,1)])
#         thisLabel2[ind2] = 1
#         tpr_mat2 = np.multiply(thisLabel2,trueLabels)
#         tpr2[0,thd_i] = np.sum(tpr_mat2)/nTrueLabels
#         fpr_mat2 = np.multiply(thisLabel2, 1-trueLabels)
#         fpr2[0,thd_i] = np.sum(fpr_mat2)/nFalseLabels
        
#         #print fpr
#        # print tpr  
#     if(plotROC == 'true'):
#         plt.xlim(min(fpr[0,:]), max(fpr[0,:]))
#         plt.xlabel('False positive Rate', fontsize=18)
#         plt.ylabel('True Positive Rate', fontsize=16)
#         plt.plot(fpr[0,:],tpr[0,:], 'b.-')
#         plt.plot(fpr2[0,:],tpr2[0,:], 'r.-')
        
#         plt.show()

#     return fpr,tpr,thdArr