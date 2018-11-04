
import numpy as np
import tensorflow as tf
from tqdm import tqdm_notebook
import pandas as pd
from keras.utils import np_utils
import csv
import math
from sklearn.cluster import KMeans

maxAcc = 0.0
maxIter = 0
C_Lambda = 0.03
TrainingPercent = 80
ValidationPercent = 10
TestPercent = 10
PHI = []
IsSynthetic = False

# This method reads data from feature file and saves data in dictionary object.
def GenerateFeatureData(filePath, data):    
    dataMatrix = [] 
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(column)
            dataMatrix.append(dataRow)
    dataDict = {}
    for i in range(1,len(dataMatrix)):
        if data == 'HUMO' : 
             img = dataMatrix[i][1]
             featureList = dataMatrix[i][2:]
        else:
             img = dataMatrix[i][0]
             featureList = dataMatrix[i][1:]
        list = []
        for i in range(len(featureList)):
            list.append(int(featureList[i]))
        
        dataDict[img]=list
    
    return dataDict


# This method concats features for same pair data 
def samePairDataConcat(filePath,featureDataDict,dataSet):    
    dataMatrix = [] 
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(column)
            dataMatrix.append(dataRow)
    dataList = []
    dataLength = 0
    if dataSet == 'HUMO':
        dataLength =  len(dataMatrix)
    else:
        dataLength = 2000         
    for i in range(1,len(dataMatrix)):
         if i<= dataLength:
             row = []
             img1 = dataMatrix[i][0]
             row.append(img1)
             dataRow1 = featureDataDict[img1]
             img2 = dataMatrix[i][1]
             row.append(img2)
             dataRow2 = featureDataDict[img2]
             dataRowConcat = dataRow1+dataRow2+[1]
             dataList.append(dataRowConcat)
    dataMatrix = np.asarray(dataMatrix)
    
    return dataList

# This method subtracts features for same pair data
def samePairDataSubtract(filePath,featureDataDict,dataSet):    
    dataMatrix = [] 
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(column)
            dataMatrix.append(dataRow)
    dataList = []
    if dataSet == 'HUMO':
        dataLength =  len(dataMatrix)
    else:
        dataLength = 2000
    for i in range(1,dataLength):
         if i<= dataLength:
             row = []
             img1 = dataMatrix[i][0]
             row.append(img1)
             dataRow1 = featureDataDict[img1]
             img2 = dataMatrix[i][1]
             row.append(img2)
             dataRow2 = featureDataDict[img2]
             subtract = np.abs(np.subtract(dataRow1,dataRow2))
             dataRowSubtract = subtract.tolist()+[1]
             dataList.append(dataRowSubtract)
    dataMatrix = np.asarray(dataMatrix)
    #print(dataDict)
    
    return dataList

# This method concats features for different pair data
def differentPairDataConcat(filePath,featureDataDict,samePairDataList,dataSet):    
    dataMatrix = [] 
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(column)
            dataMatrix.append(dataRow)
    dataList = []
    #print(len(samePairDataList))
    if dataSet == 'HUMO':
        dataLength =  len(samePairDataList)
        print('Hey')
    else:
        dataLength = 2000
    for i in range(1,len(dataMatrix)):
         if i<= dataLength:
             row = []
             img1 = dataMatrix[i][0]
             row.append(img1)
             dataRow1 = featureDataDict[img1]
             img2 = dataMatrix[i][1]
             row.append(img2)
             dataRow2 = featureDataDict[img2]
             dataRowConcat = dataRow1+dataRow2+[0]
             dataList.append(dataRowConcat)
    dataMatrix = np.asarray(dataMatrix)
    
    return dataList

# This method subtracts features for different pair data
def differentPairDataSubtract(filePath,featureDataDict,samePairDataList,dataSet):    
    dataMatrix = [] 
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(column)
            dataMatrix.append(dataRow)
    dataList = []
    
    if dataSet == 'HUMO':
        dataLength =  len(samePairDataList)
    else:
        dataLength = 2000
    for i in range(1,dataLength):
         if i<= dataLength:
             row = []
             img1 = dataMatrix[i][0]
             row.append(img1)
             dataRow1 = featureDataDict[img1]
             img2 = dataMatrix[i][1]
             row.append(img2)
             dataRow2 = featureDataDict[img2]
             subtract = np.abs(np.subtract(dataRow1,dataRow2))
             dataRowSubtract = subtract.tolist()+[0]
             dataList.append(dataRowSubtract)
    dataMatrix = np.asarray(dataMatrix)
   
    return dataList

def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t= 0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))

# This method generates target vector from raw data.
def getTargetVector(humanObservedDataMatrix):
    w = humanObservedDataMatrix.shape[1]
    t = humanObservedDataMatrix[...,w-1]
    return t

# This method generates training target vector
def GenerateTrainingTarget(rawTraining,TrainingPercent = 80):
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
    t           = rawTraining[:TrainingLen]
    t = np.array(t)
    return t

# This method generates training data 
def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):
    T_len = int(math.ceil(rawData.shape[0]*0.01*TrainingPercent))
    d2 = rawData[:T_len]
    return d2

# This method generates validation / testing data.
def GenerateValData(rawData, ValPercent, TrainingCount):
    
    valSize = int(math.ceil(rawData.shape[0]*ValPercent*0.01))
    V_End = TrainingCount + valSize
    dataMatrix = rawData[TrainingCount:V_End-1] 
    return dataMatrix

# This method generates validation / testing target vector.
def GenerateValTargetVector(rawData, ValPercent, TrainingCount):
    
    valSize = int(math.ceil(rawData.shape[0]*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t = rawData[TrainingCount:V_End-1]
    return t

def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic):
    
    BigSigma    = np.zeros((Data.shape[1],Data.shape[1]))      
    varVect     = []
    for i in range(0,Data.shape[1]):
        vct = []
        for j in range(0,Data.shape[0]):
            vct.append(Data[j,i])    
        varVect.append(np.var(vct))
    

    for j in range(Data.shape[1]):
        BigSigma[j][j] = varVect[j]
    return BigSigma

def GetScalar(DataRow,MuRow, BigSigInv):  
    R = np.subtract(DataRow,MuRow)
    T = np.dot(BigSigInv,np.transpose(R))  
    L = np.dot(R,T)
    return L

def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

def GetPhiMatrix(Data, MuMatrix, BigSigma):
    #DataT = np.transpose(Data)
    #TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))         
    PHI = np.zeros((Data.shape[0],len(MuMatrix)))
    #print(BigSigma.shape) 
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,Data.shape[0]):
            PHI[R][C] = GetRadialBasisOut(Data[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    return PHI

# removing zero var columns
def RemoveVarZeroColumns(Data):
    varNonZeroData = Data
    columnsDelete = []
    for i in range(0,Data.shape[1]):
        vct = []
        for j in range(0,Data.shape[0]):
            vct.append(Data[j,i])
        if np.var(vct) == 0:
            columnsDelete.append(i)
    varNonZeroData = np.delete(varNonZeroData, columnsDelete, axis = 1)
    return varNonZeroData

def addDataSameDiffPairLinearRegression(samePairDataList,differentPairDataList):
    rawData = samePairDataList + differentPairDataList
    rawDataMatrix = np.matrix(rawData)
    np.random.shuffle(rawDataMatrix)
    print('.....................')
    #print(rawDataMatrix)
    return rawDataMatrix
# removing zero var columns
def removeZeroVarColumns(rawDataMatrix):
    rawDataMatrix = RemoveVarZeroColumns(rawDataMatrix)
    return rawDataMatrix

# add two data sets i.e .sampe pair data set and different pair data set and shuffles them
def addDataSameDiffPair(samePairDataList,differentPairDataList):
    rawData = samePairDataList + differentPairDataList
    rawDataMatrix = np.array(rawData)
    np.random.shuffle(rawDataMatrix)
    return rawDataMatrix


# removing target values from data matrix that we have appended while generating data sets.
def removeTargetValues(rawDataMatrix):
    w = rawDataMatrix.shape[1]-1
    rawDataMatrix = rawDataMatrix[...,:w]
    return rawDataMatrix

# applies sigmoid function on input data 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Add bias for logistic regression 
def addBias(rawDataMatrix):
     temp = rawDataMatrix
     rawDataMatrix = np.ones((rawDataMatrix.shape[0], rawDataMatrix.shape[1]+1))
     rawDataMatrix[:, :-1] = temp

     # temp = np.ones(rawDataMatrix.shape[0],rawDataMatrix.shape[1]+1)
     # temp[:,0:rawDataMatrix.shape[1]] = rawDataMatrix
     return temp

# weight updates for logistic regression
def updateWeights(X, y):
    W = np.zeros(X.shape[1])
    lr = 0.01
    for i in range(X.shape[0]):
        z = np.dot(X,W)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / y.size
        W -= lr * gradient
   
    return W

# this method returns accuracy of models            
def getAccuracy(predictedValues, targetValues):
    accuracy = 0
    failure = 0
    for i in range(0,predictedValues.size):
        if predictedValues[i] == targetValues[i]:
            accuracy = accuracy + 1
        else:
            failure = failure + 1
    #print(failure)
    return accuracy/predictedValues.size * 100    

# initialize weights for neural networks.
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))

# defining the Neural Networks model.
def defineModel(M): 
    inputTensor  = tf.placeholder(tf.float32, [None, M])
    outputTensor = tf.placeholder(tf.float32, [None, 1])

    NUM_HIDDEN_NEURONS_LAYER_1 = 200
    LEARNING_RATE = 0.05

    input_hidden_weights  = init_weights([M, NUM_HIDDEN_NEURONS_LAYER_1])
    hidden_output_weights = init_weights([NUM_HIDDEN_NEURONS_LAYER_1, 1])

    hidden_layer = tf.nn.sigmoid(tf.matmul(inputTensor, input_hidden_weights))
    output_layer = tf.matmul(hidden_layer, hidden_output_weights)

    error_function = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_layer, labels=outputTensor))
    training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(error_function)
    prediction = tf.sigmoid(output_layer)

    return prediction, training, inputTensor, outputTensor

def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI))
    ##print ("Test Out Generated..")
    return Y

# Generates PHI matrices
def generatePHIMatrices(M,TrainingPercent,trainingDataMatrix,testingDataMatrix,validationDataMatrix):

    ErmsArr = []
    AccuracyArr = []

    kmeans = KMeans(n_clusters=M, random_state=0).fit(trainingDataMatrix)
    Mu = kmeans.cluster_centers_
    BigSigma = GenerateBigSigma(trainingDataMatrix, Mu, TrainingPercent,IsSynthetic)
    #print(BigSigma)
    TRAINING_PHI = GetPhiMatrix(trainingDataMatrix, Mu, BigSigma)
    print(TRAINING_PHI.shape)
    TEST_PHI     = GetPhiMatrix(testingDataMatrix, Mu, BigSigma)
    print(TEST_PHI.shape)
    VAL_PHI      = GetPhiMatrix(validationDataMatrix, Mu, BigSigma)
    print(VAL_PHI.shape)
    return TRAINING_PHI, TEST_PHI, VAL_PHI

# Training Neural Networks model.
def trainModel(prediction,training, inputTensor, outputTensor,trainingData,trainingTarget,testingData):
    NUM_OF_EPOCHS = 4000
    BATCH_SIZE = 128

    training_accuracy = []

    with tf.Session() as sess:
    
        tf.global_variables_initializer().run()
    
        for epoch in tqdm_notebook(range(NUM_OF_EPOCHS)):
        
        
            
            # Start batch training
            for start in range(0, len(trainingData), BATCH_SIZE):
                end = start + BATCH_SIZE
                sess.run(training, feed_dict={inputTensor: trainingData[start:end], 
                                          outputTensor: trainingTarget[start:end]})
            # Training accuracy for an epoch
            training_accuracy.append(np.mean(np.argmax(trainingTarget, axis=1) ==
                             sess.run(prediction, feed_dict={inputTensor: trainingData,
                                                             outputTensor: trainingTarget})))
        # Testing
        # this line of code gives predicted output for testing data based on above trained model
        predictedTestValues = sess.run(prediction, feed_dict={inputTensor: testingData})

        return predictedTestValues


def calculateErms(M,trainingDataMatrix,trainingTargetVector,TRAINING_PHI,VAL_PHI,validationTargetVector,TEST_PHI,testingTargetVector):

    W = np.ones((M,))
    #print(W.shape)
    W_Now        = np.dot(220, W)
    #print(W_Now)
    La           = 2
    learningRate = 0.01
    L_Erms_Val   = []
    L_Erms_TR    = []
    L_Erms_Test  = []
    W_Mat        = []
    L_Erms_Accuracy = []
    #print(trainingTargetVector.shape)
    for i in range(0,1000):
        print(i)
        #print ('---------Iteration: ' + str(i) + '--------------')
        Delta_E_D     = -np.dot((trainingTargetVector[i,0] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
        La_Delta_E_W  = np.dot(La,W_Now)
        Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
        Delta_W       = -np.dot(learningRate,Delta_E)
        W_T_Next      = W_Now + Delta_W
        W_Now         = W_T_Next
    
        #-----------------TrainingData Accuracy---------------------#
        TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
        #print(TR_TEST_OUT)
        Erms_TR       = GetErms(TR_TEST_OUT,trainingTargetVector)
        L_Erms_TR.append(float(Erms_TR.split(',')[1]))
        #L_Erms_Accuracy.append(float(Erms_TR.split(',')[0]))
    
        #-----------------ValidationData Accuracy---------------------#
        VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
        #print(VAL_TEST_OUT)
        Erms_Val      = GetErms(VAL_TEST_OUT,validationTargetVector)
        L_Erms_Val.append(float(Erms_Val.split(',')[1]))
    
        #-----------------TestingData Accuracy---------------------#
        TEST_OUT  = GetValTest(TEST_PHI,W_T_Next)
        #print(TEST_OUT)
        Erms_Test = GetErms(TEST_OUT,testingTargetVector)
        L_Erms_Test.append(float(Erms_Test.split(',')[1]))
    print ('----------Gradient Descent Solution--------------------')
    print ("M = 15 \nLambda  = 0.0001\neta=0.01")
    print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
    print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
    print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))


def linearRegression(samePairDataList,differentPairDataList):
     rawDataMatrix = addDataSameDiffPairLinearRegression(samePairDataList,differentPairDataList)

     rawDataMatrix = removeZeroVarColumns(rawDataMatrix)
     TargetVector = getTargetVector(rawDataMatrix)
     # Prepare Training Data
     trainingDataMatrix = GenerateTrainingDataMatrix(rawDataMatrix,TrainingPercent)
     trainingTargetVector = GenerateTrainingTarget(TargetVector,TrainingPercent)

     # Prepare Validation Data
     validationDataMatrix = GenerateValData(rawDataMatrix,ValidationPercent, trainingDataMatrix.shape[0])
     validationTargetVector = GenerateValTargetVector(TargetVector,ValidationPercent, trainingDataMatrix.shape[0])

     # Prepare Testing Data
     testingDataMatrix = GenerateValData(rawDataMatrix,ValidationPercent, trainingDataMatrix.shape[0]+validationDataMatrix.shape[0])
     testingTargetVector = GenerateValTargetVector(TargetVector,ValidationPercent, trainingDataMatrix.shape[0]+validationDataMatrix.shape[0])
     M = 10
     TRAINING_PHI, TEST_PHI, VAL_PHI = generatePHIMatrices(M,TrainingPercent,trainingDataMatrix,testingDataMatrix,validationDataMatrix)

     print("............")
     print(TRAINING_PHI.shape)
     calculateErms(M,trainingDataMatrix,trainingTargetVector,TRAINING_PHI,VAL_PHI,validationTargetVector,TEST_PHI,testingTargetVector)


def logisticRegression(TargetVector,rawDataMatrix):
     
     rawDataMatrix = addBias(rawDataMatrix)
     trainingDataMatrix = GenerateTrainingDataMatrix(rawDataMatrix,TrainingPercent)
     trainingTargetVector = GenerateTrainingTarget(TargetVector,TrainingPercent)

     # Prepare Validation Data
     validationDataMatrix = GenerateValData(rawDataMatrix,ValidationPercent, trainingDataMatrix.shape[0])
     validationTargetVector = GenerateValTargetVector(TargetVector,ValidationPercent, trainingDataMatrix.shape[0])

     # Prepare Testing Data
     testingDataMatrix = GenerateValData(rawDataMatrix,ValidationPercent, trainingDataMatrix.shape[0]+validationDataMatrix.shape[0])
     testingTargetVector = GenerateValTargetVector(TargetVector,ValidationPercent, trainingDataMatrix.shape[0]+validationDataMatrix.shape[0])

     # find updated weights
     W = updateWeights(trainingDataMatrix,trainingTargetVector)
     print('//.......... Logistic regression ........//')
     validationPredictValues = sigmoid(np.dot(validationDataMatrix, W))
     validation_accuracy = getAccuracy(np.round(validationPredictValues), validationTargetVector) 
     print("Validation Accuracy: ")
     print(validation_accuracy)


     testingPredictedValues = sigmoid(np.dot(testingDataMatrix, W))
     testing_accuracy = getAccuracy(np.round(testingPredictedValues), testingTargetVector)
     print("Testing Accuracy: ")
     print(testing_accuracy) 


def neuralNetworks(TargetVector,rawDataMatrix):
     # Prepare Training Data
     trainingDataMatrix = GenerateTrainingDataMatrix(rawDataMatrix,TrainingPercent)
     trainingTargetVector = GenerateTrainingTarget(TargetVector,TrainingPercent)
     trainingTargetVector = trainingTargetVector.reshape(trainingTargetVector.shape[0],1)

     # Prepare Validation Data
     validationDataMatrix = GenerateValData(rawDataMatrix,ValidationPercent, trainingDataMatrix.shape[0])
     validationTargetVector = GenerateValTargetVector(TargetVector,ValidationPercent, trainingDataMatrix.shape[0])
     #validationTargetVector = validationTargetVector.reshape(validationTargetVector.shape[0],1)

     # Prepare Testing Data
     testingDataMatrix = GenerateValData(rawDataMatrix,ValidationPercent, trainingDataMatrix.shape[0]+validationDataMatrix.shape[0])
     testingTargetVector = GenerateValTargetVector(TargetVector,ValidationPercent, trainingDataMatrix.shape[0]+validationDataMatrix.shape[0])

     prediction_model,training, inputTensor, outputTensor = defineModel(trainingDataMatrix.shape[1])
     validationPredictValues  = trainModel(prediction_model,training, inputTensor, outputTensor,trainingDataMatrix,trainingTargetVector,validationDataMatrix)
     testingPredictValues  = trainModel(prediction_model,training, inputTensor, outputTensor,trainingDataMatrix,trainingTargetVector,testingDataMatrix)
     validation_accuracy = getAccuracy(np.round(validationPredictValues), validationTargetVector)
     testing_accuracy = getAccuracy(np.round(testingPredictValues), testingTargetVector) 
     print("//.......... Neural Networks ..............//")
     print("Validation Accuracy: ")
     print(validation_accuracy)
     print('\n')
     print("Testing Accuracy: ")
     print(testing_accuracy)

print("\n.............. Human Observed Data Subtract Setting .........................")
print('\n')
featureDataDict = GenerateFeatureData('HumanObserved-Features-Data.csv','HUMO')
samePairDataList = samePairDataSubtract('same_pairs.csv',featureDataDict,'HUMO')
differentPairDataList = differentPairDataSubtract('diffn_pairs.csv',featureDataDict,samePairDataList,len(samePairDataList))
rawDataMatrix = addDataSameDiffPair(samePairDataList,differentPairDataList)
TargetVector = getTargetVector(rawDataMatrix)
rawDataMatrix = removeTargetValues(rawDataMatrix)
linearRegression(samePairDataList,differentPairDataList)
logisticRegression(TargetVector,rawDataMatrix)
neuralNetworks(TargetVector,rawDataMatrix)
print("\n.............. Human Observed Data Concat Setting .........................")
print('\n')
featureDataDict = GenerateFeatureData('HumanObserved-Features-Data.csv','HUMO')
print(len(featureDataDict))
samePairDataList = samePairDataConcat('same_pairs.csv',featureDataDict,'HUMO')
differentPairDataList = differentPairDataConcat('diffn_pairs.csv',featureDataDict,samePairDataList,'HUMO')
rawDataMatrix = addDataSameDiffPair(samePairDataList,differentPairDataList)
TargetVector = getTargetVector(rawDataMatrix)
rawDataMatrix = removeTargetValues(rawDataMatrix)
linearRegression(samePairDataList,differentPairDataList)
logisticRegression(TargetVector,rawDataMatrix)
neuralNetworks(TargetVector,rawDataMatrix)
print("\n.............. GSC Data Concat Setting .........................")
print('\n')
featureDataDict = GenerateFeatureData('GSC-Features.csv','GSC')
samePairDataList = samePairDataConcat('same_pairs_GSC.csv',featureDataDict,'GSC')
differentPairDataList = differentPairDataConcat('diffn_pairs_GSC.csv',featureDataDict,samePairDataList,'GSC')
rawDataMatrix = addDataSameDiffPair(samePairDataList,differentPairDataList)
TargetVector = getTargetVector(rawDataMatrix)
rawDataMatrix = removeTargetValues(rawDataMatrix)
linearRegression(samePairDataList,differentPairDataList)
logisticRegression(TargetVector,rawDataMatrix)
neuralNetworks(TargetVector,rawDataMatrix)
print("\n.............. GSC Data Subtract Setting .........................")
print('\n')
featureDataDict = GenerateFeatureData('GSC-Features.csv','GSC')
samePairDataList = samePairDataSubtract('same_pairs_GSC.csv',featureDataDict,'GSC')
differentPairDataList = differentPairDataSubtract('diffn_pairs_GSC.csv',featureDataDict,samePairDataList,'GSC')
rawDataMatrix = addDataSameDiffPair(samePairDataList,differentPairDataList)
TargetVector = getTargetVector(rawDataMatrix)
rawDataMatrix = removeTargetValues(rawDataMatrix)
linearRegression(samePairDataList,differentPairDataList)
logisticRegression(TargetVector,rawDataMatrix)
neuralNetworks(TargetVector,rawDataMatrix)

