import pickle
from ai_model import predict
import os
import math
import numpy as np


#Initializing lists with the features' values. 
counties = ['agder', 'innlandet' , 'møre og romsdal', 'nordland', 'oslo', 'rogaland',
            'troms og finnmark', 'trøndelag', 'vestfold og telemark', 'vestland', 'viken']

ages = ['16-19', '20-24', '25-29', '30-34', '35-39', '40-44',
        '45-49', '50-54', '55-59', '60-64', '65-69']

diagnosis = ['cardiovascular diseases','muscle/skeleton disorders', 'mental disorders', 'pregnancy disorders',
            'diseases in the digestive organs', 'diseases in the respiratory tract', 'diseases in the nervous system']

gender = ['male', 'female']

allValues = counties + ages + diagnosis + gender


"""
Function:
        Gets prediction for all personas in the dataset. A persona corresponds to a distinct combination of feature values.
        Example of a persona: ['agder', '16-19', 'mental disorders', 'male']       
Args: 
        None

Returns: 
        None. (Dictionary with all personas and their corresponding predictions are 
                instead saved as a pickle-file)
"""
def getAndSaveAllPredictions():
    predictions = {}
    for c in counties:
        for a in ages:
            for d in diagnosis:
                for g in gender:
                    result = predict([c, a, d, g])
                    predictions[c, a, d, g] = result
    #print(predictions)
    cd = os.getcwd()
    pickle.dump(predictions, open(cd + "\\data\\allPredictions.sav", 'wb'))


"""
Function:
        Gets all personas not containing some feature value. 
Args: 
        val (string): some feaure value, e.g 'agder' or 'female'
        allVals (dictionary): containing all personas as keys and their predictions as values. 

Returns: 
        index (int): index of feature value (val) in personas. 
        other (list): personas not containing the given feature value (val).
"""
def getOtherPredictions(val, allVals):
    other = []
    index = 0
    if val in counties:
        index = 0
    elif val in ages:
        index = 1
    elif val in diagnosis:
        index = 2
    elif val in gender:
        index = 3

    #Iterating through predictions and finding those not containing val
    for key in allVals.keys():
        if val not in key:
            other.append(key)
    
    return index, other


"""
Function:
        Calculates shapley value for each feature value. 
Args: 
        None

Returns: 
        None. (dictionary with feature values and their corresponding shapley values
                are instead saved as a pickle-file)
"""
def calculateShapleyvalues():   
    shapleyValues = {}

    #Fetching predictions
    cd = os.getcwd()
    with open(cd + "\\data\\allPredictions.sav", 'rb') as f:
        predictions = pickle.load(f)

    #Calculating shapley-value for each feature-value
    for val in allValues:
        index, S = getOtherPredictions(val, predictions)
        v = 0
        for pred in S:
            predWithVal = list(pred)
            predWithVal[index] = val
            v += (((((math.factorial(len(pred))) * math.factorial((len(allValues) - len(pred) - 1))) / math.factorial(len(allValues))) 
                    * abs((predictions[tuple(predWithVal)] - predictions[pred]))))

        shapleyValues[val] = v
    
    #Storing shapley-values
    cd = os.getcwd()
    pickle.dump(shapleyValues, open(cd + "\\data\\shapleyValues.sav", 'wb'))


"""
Function:
        Prints out each feature value with their corresponding shapley value.
        This function was only used as a tool to interpret the values and was only for the coders benefit. 
Args: 
        None

Returns: 
        None. (Prints to console) 
"""
def interpretShapleyValues():
    cd = os.getcwd()   
    with open(cd + "\\data\\shapleyValues.sav", 'rb') as f:
        sv = pickle.load(f) 

    for key in sv:
        print("----------------")
        print("feature: ", key)
        print("shapleyValue: ", sv[key])  


"""
Function:
        Calculates average feature value for the features county, diagnosis and gender.
Args: 
        None. 

Returns: 
        results (list): tuples for each feature value in age, 
                        and a list containing its corresponding shapley value and the average shapley values.
        Example of tuple in results: ('16-19', [0.0007550476960932934, 0.0025742888924201583, 0.002095214553501534, 0.000818075345327839])
"""
def findAverage():
    #Fetching shapleyValues
    cd = os.getcwd()
    with open(cd + "\\data\\shapleyValues.sav", 'rb') as f:
        shapleyValues = pickle.load(f)

    C, D, G, A = [], [], [], []

    for val in shapleyValues:
        if val in counties:
            C.append(shapleyValues[val])
        elif val in diagnosis:
            D.append(shapleyValues[val])
        elif val in gender:
            G.append(shapleyValues[val])
        else:
            A.append((val, shapleyValues[val]))
    
    C_average = np.average(C)
    D_average = np.average(D)
    G_average = np.average(G)
    results = [(a[0], [ C_average, a[1], D_average, G_average]) for a in A]

    return results


"""
Function:
        normalizes lists found in findAverage(), this way they will sum to 1 and can be used as percentages. 
Args: 
        values (list): tuples for each feature value in age, 
                        and a list containing its corresponding shapley value and the average shapley values.  
Returns: 
        None. (Prints to console) 
"""
def normalize(values):
    normalizedValues = [(v[0], np.round([float(i)/sum(v[1]) for i in v[1]], 2)) for v in values]

    for nv in normalizedValues:
        while sum(nv[1]) < 1.0:
            lowest = list(nv[1]).index(min(nv[1]))
            nv[1][lowest] += 0.01

        print("----------------")
        print("Alder: ", nv[0])
        print("a: ",  nv[1][1] )
        print("c: ",  nv[1][0] )
        print("g: ",  nv[1][3] )
        print("d: ",  nv[1][2] )
        print("sum: ", sum(nv[1]))

   

def run():
    """
    Sequential order of function-calls to calculate sets of shapley values.
    """
    #getAndSaveAllPredictions()
    #calculateShapleyvalues()
    #results = findAverage()
    #normalize(results)

    """
    Prints out pair of feature values and their shapley values, only for sanity check.
    """
    #interpretShapleyValues()


if __name__ == "__main__":
    run()

