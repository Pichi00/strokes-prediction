#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random as rd
import pandas as pd
import seaborn as sns
import matplotlib as plt

strokes = pd.read_csv('healthcare-dataset-stroke-data.csv')
strokes.info()
strokes.describe()
strokes.head()

sns.violinplot(y='stroke', x='work_type', data=strokes, inner='quartile')

sns.pairplot(strokes, hue='stroke', markers='+')

def ToNumData(X):
    for col in X.columns:
        if col == "gender":
            for row_nr in range(len(X[col])):
                if(X[col].iloc[row_nr] == "Male"):
                    X[col].iloc[row_nr] = 0
                elif (X[col].iloc[row_nr] == "Female"):
                    X[col].iloc[row_nr] = 1 
                elif (X[col].iloc[row_nr] == "Other"):
                    X[col].iloc[row_nr] = -1 
            
        elif col == "ever_married":
            for row_nr in range(len(X[col])):
                if(X[col].iloc[row_nr] == "Yes"):
                    X[col].iloc[row_nr] = 1
                elif (X[col].iloc[row_nr] == "No"):
                    X[col].iloc[row_nr] = 0 
            
        elif col == "Residence_type":
            for row_nr in range(len(X[col])):
                if(X[col].iloc[row_nr] == "Rural"):
                    X[col].iloc[row_nr] = 1
                elif (X[col].iloc[row_nr] == "Urban"):
                    X[col].iloc[row_nr] = 0 
             
        elif col == "smoking_status":
            for row_nr in range(len(X[col])):
                if(X[col].iloc[row_nr] == "formerly smoked"):
                    X[col].iloc[row_nr] = 1
                elif (X[col].iloc[row_nr] == "never smoked"):
                    X[col].iloc[row_nr] = 0  
                elif (X[col].iloc[row_nr] == "smokes"):
                    X[col].iloc[row_nr] = 2 
                elif (X[col].iloc[row_nr] == "Unknown"):
                    X[col].iloc[row_nr] = -1  
        
        elif col == "work_type":
            for row_nr in range(len(X[col])):
                if(X[col].iloc[row_nr] == "Never_worked"):
                    X[col].iloc[row_nr] = 0
                elif (X[col].iloc[row_nr] == "Govt_job"):
                    X[col].iloc[row_nr] = 1  
                elif (X[col].iloc[row_nr] == "Private"):
                    X[col].iloc[row_nr] = 2 
                elif (X[col].iloc[row_nr] == "children"):
                    X[col].iloc[row_nr] = -1  
                elif (X[col].iloc[row_nr] == "Self-employed"):
                    X[col].iloc[row_nr] = 3  
                        
    return X

#Dropping rows where bmi is NaN
strokes.dropna(subset= ["bmi"], inplace=True)

#Changing non-numeric data to numeric types
strokes = ToNumData(strokes)
strokes["gender"] = pd.to_numeric(strokes["gender"])
strokes["ever_married"] = pd.to_numeric(strokes["ever_married"])
strokes["Residence_type"] = pd.to_numeric(strokes["Residence_type"])
strokes["smoking_status"] = pd.to_numeric(strokes["smoking_status"])
strokes["work_type"] = pd.to_numeric(strokes["work_type"])

strokesTrain = strokes.sample(frac=0.7, random_state=1) 
strokesVal = strokes.drop(strokesTrain.index)

yTrain = strokesTrain["stroke"]
xTrain = strokesTrain.drop("stroke", axis = 1).drop("id", axis = 1)

yVal = strokesVal["stroke"]
xVal = strokesVal.drop("stroke", axis = 1).drop("id", axis = 1)

means = strokesTrain.groupby(["stroke"]).mean() 
var = strokesTrain.groupby(["stroke"]).var() 
classes = np.unique(strokesTrain["stroke"].tolist())
prior = (strokesTrain.groupby(["stroke"]).count()/len(strokesTrain)).iloc[:,0]

def Normal(n, mu, var):
    sd = np.sqrt(var)
    pdf = (np.e ** (-0.5 * ((n - mu)/sd) ** 2)) / (sd * np.sqrt(2 * np.pi))    
    return pdf # pdf - probability density function

def Predict(X):
    Predictions = []
    
    for i in X.index: # Loop through each instances
        ClassLikelihood = []
        instance = X.loc[i]
        
        for cls in classes: # Loop through each class
            
            FeatureLikelihoods = []
            FeatureLikelihoods.append(np.log(prior[cls])) # Append log prior of class 'cls'
            
            for col in X.columns: # Loop through each feature
                
                data = instance[col]
                
                mean = means[col].loc[cls] # Find the mean of column 'col' that are in class 'cls'
                variance = var[col].loc[cls] # Find the variance of column 'col' that are in class 'cls'
                
                Likelihood = Normal(data, mean, variance)
                
                if Likelihood != 0:
                    Likelihood = np.log(Likelihood) # Find the log-likelihood evaluated at x
                else:
                    Likelihood = 1/len(X) 
                
                FeatureLikelihoods.append(Likelihood)
                
            TotalLikelihood = sum(FeatureLikelihoods) # Calculate posterior
            ClassLikelihood.append(TotalLikelihood)
            
        MaxIndex = ClassLikelihood.index(max(ClassLikelihood)) # Find largest posterior position
        Prediction = classes[MaxIndex]
        Predictions.append(Prediction)
        
    return Predictions

PredictTrain = Predict(xTrain)
PredictVal = Predict(xVal)

def Accuracy(y, prediction):
    
    # Function to calculate accuracy
    y = list(y)
    prediction = list(prediction)
    score = 0
    
    for i, j in zip(y, prediction):
        if i == j:
            score += 1
            
    return score / len(y)

round(Accuracy(yTrain, PredictTrain), 5)

round(Accuracy(yVal, PredictVal), 5)

def ColumnsSelection(y,X):
    for col in X.columns:
        x = X.drop(col, axis = 1)
        prediction = Predict(x)
        print(f"Accuracy without column {col}: {str(round(Accuracy(y, prediction)*100, 5))}%")

ColumnsSelection(yTrain,xTrain)

ColumnsSelection(yVal,xVal)

def colSelection(y,X):

    i = 1
    cols_and_acc = dict()
    for col1 in X.columns:
        x1 = X.drop(col1, axis = 1)
        for col2 in x1.columns:
            x2 = x1.drop(col2, axis = 1)
            prediction = Predict(x2)
            cols_and_acc[str(col1 + " and " + col2)] = round(Accuracy(y, prediction), 5)
            i+=1
    print(f"Accuracy without columns {max(cols_and_acc, key=cols_and_acc.get)}: {str(cols_and_acc[max(cols_and_acc, key=cols_and_acc.get)])}")

colSelection(yVal,xVal)

