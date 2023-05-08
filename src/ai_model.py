from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import os
import pickle


"""
Function:
        Reads in and preprocesses data
Args: 
        None
Returns: 
        training_X (list): List of features used to train model
        training_Y (list): List of labels used to train model
        test_X     (list): List of features used to test model
        test_Y     (list): List of labels used to test model
"""
def readAndProcessData():
    #Fetching current working directory
    cd = os.getcwd()
 
    #Using Pandas to get data from excel file
    dataframe = pd.ExcelFile(cd + "\\data\\sickleaveData.xlsx")


    #Reading each sheet
    df1 = pd.read_excel(dataframe, 'Fylke')[24:] #Omitting unwanted rows
    df2 = pd.read_excel(dataframe, 'Alder')[23:] #Omitting unwanted rows
    df3 = pd.read_excel(dataframe, 'Diagnose')[22:] #Omitting unwanted rows

    #Removing rows that conatain NaN-values
    df1 = df1.dropna()
    df2 = df2.dropna()
    df3 = df3.dropna() 

    #Transposing, changing headers, and dropping unwanted columns
    df1 = df1.T
    df1_new_header = df1.iloc[0]
    df1 = df1[1:]
    df1.columns = df1_new_header
    df1 = df1.drop(columns=["I alt", "Kvinner", "Menn"])

    df2 = df2.T
    df2_new_header = df2.iloc[0]
    df2 = df2[1:]
    df2.columns = df2_new_header
    df2 = df2.drop(columns=["I alt", "Kvinner", "Menn"])

    df3 = df3.T
    df3_new_header = df3.iloc[0]
    df3 = df3[1:]
    df3.columns = df3_new_header
    df3 = df3.drop(columns=["I alt", "Kvinner", "Menn", "Ukjent", "Allment og uspesifisert", "Andre lidelser"])


    #Splitting by sex/gender
    df1Women = df1.iloc[:, :12]
    df1Men = df1.iloc[:, 12:]

    df2Women = df2.iloc[:, :11]
    df2Men = df2.iloc[:, 11:]
     
    df3Women = df3.iloc[:, :7]
    df3Men = df3.iloc[:, 7:]


    #Summing all rows and calculating mean
    df1Men = df1Men.sum()
    df1Women = df1Women.sum()
    df2Men = df2Men.sum()
    df2Women = df2Women.sum()
    df3Men = df3Men.sum()
    df3Women = df3Women.sum()
    
    allDataframes = [df1Women,  df2Women, df3Women, df1Men, df2Men,  df3Men]
    
    for elem in allDataframes:
        for col in elem.keys():
            elem[col] = elem[col]/4
 

    #Combining data to generate personas, and setting mean-value of attribute-values as labels
    women = allDataframes[:3]
    men = allDataframes[3:]
    temporaryData = {'Fylke':[], 'Alder':[], 'Diagnose':[], 'Sex':[], 'labels':[]}
    for fylke in women[0].keys():
        for alder in women[1].keys():
            for diagnose in women[2].keys():
                meanValue = int(((women[0][fylke] + women[1][alder] + women[2][diagnose])/3))
                meanValueInWeeks = int(meanValue/7)

                #Appending Attributes and corresponding label to dictionary.
                temporaryData['Fylke'].append(str(fylke))
                temporaryData['Alder'].append(str(alder))
                temporaryData['Diagnose'].append(str(diagnose))
                temporaryData['Sex'].append('woman')
                temporaryData['labels'].append(meanValueInWeeks)
    
    for fylke in men[0].keys():
        for alder in men[1].keys():
            for diagnose in men[2].keys():
                meanValue = int(((men[0][fylke] + men[1][alder] + men[2][diagnose])/3))  
                meanValueInWeeks = int(meanValue/7) 

                #Appending Attributes and corresponding label to dictionary.
                temporaryData['Fylke'].append(str(fylke))
                temporaryData['Alder'].append(str(alder))
                temporaryData['Diagnose'].append(str(diagnose))
                temporaryData['Sex'].append('man')                
                temporaryData['labels'].append(meanValueInWeeks)


    #Converting features from string to numbers.
    tempDf = pd.DataFrame(temporaryData, columns=['Fylke', 'Alder', 'Diagnose', 'Sex', 'labels'])
    tempDf = pd.get_dummies(tempDf, columns=['Fylke', 'Alder' , 'Diagnose', 'Sex'])

    #Too few examples with labels 2 or 9 for the model to classify these accurately, thus changing them to 3 and 8 labels.
    tempDf.loc[tempDf["labels"] == 9, "labels"] = 8
    tempDf.loc[tempDf["labels"] == 2, "labels"] = 3

    #Splitting into training and test-data. (80/20)   
    trainDf, testDf = train_test_split(tempDf, test_size=0.2, random_state=42)

    #Converting from DataFrame to-array-to-list, and seperating label and features. X contains features, and Y contains corresponding labels.
    training_X = trainDf.to_numpy().tolist()
    training_Y = [elem.pop(0) for elem in training_X]

    test_X = testDf.to_numpy().tolist()
    test_Y = [elem.pop(0) for elem in test_X]
    
    #Uncomment below to print example distribution among classes
    #print("Distribution in training set: ", np.unique(training_Y, return_counts=True))
    #print("Distribution in test set: ", np.unique(test_Y, return_counts=True))


    #Returning generated datasets. 
    return training_X, training_Y, test_X, test_Y


"""
Function:
        Trains and tests AI-model
Args: 
        training_X (list): List of features used to train the model
        training_Y (list): List of labels used to train the model
        test_X     (list): List of features used to test the model
        test_Y     (list): List of labels used to test the model

Returns: 
        None (Model is instead saved as a pickle-file)  
"""
def trainModel(training_X, training_Y, test_X, test_Y, model):

    #Training model
    trainedModel = model.fit(training_X, training_Y)

    #Evaluating models performance, based on test data, i.e data it has never seen before
    predicted = trainedModel.predict(test_X)
    correct = 0
    for i in range(len(predicted)):
        if predicted[i] == test_Y[i]:
            correct += 1

    #Printing accuracy and f1-score
    print(f"Accuracy: ", correct/len(predicted))
    print(f"f1-score: ", str(f1_score(test_Y, predicted, average=None)))

    #Saving trained model as a pickle-file
    cd = os.getcwd()
    pickle.dump(trainedModel, open(cd + "\\data\\trained_model.sav", 'wb'))


"""
Function:
        Predicts sick-leave for persona-example
Args: 
        input (list): Persona-example containing both features and label

Returns: 
        prediction[0] (int): Persona-example's predicted sick-leave 
"""
def predict(input):

    #Converting input from string to int
    result = np.zeros((1, 31), dtype=int).tolist()[0]
    features = {'agder': [0, 0], 'innlandet': [0, 1], 'møre og romsdal': [0, 2], 'nordland': [0, 3], 'oslo': [0, 4], 'rogaland': [0, 5],
                'troms og finnmark': [0, 6], 'trøndelag': [0, 7], 'vestfold og telemark': [0, 8], 'vestland': [0, 9], 'viken': [0, 10],
                '16-19': [0, 11], '20-24': [0, 12], '25-29': [0, 13], '30-34': [0, 14], '35-39': [0, 15], '40-44': [0, 16],
                '45-49': [0, 17], '50-54': [0, 18], '55-59': [0, 19], '60-64': [0, 20], '65-69': [0, 21], 'cardiovascular diseases': [0, 22],
                'muscle/skeleton disorders': [0, 23], 'mental disorders': [0, 24], 'pregnancy disorders': [0, 25], 'diseases in the digestive organs': [0, 26],
                'diseases in the respiratory tract': [0, 27], 'diseases in the nervous system': [0, 28], 'male': [0, 29], 'female': [0, 30]}

    for elem in input:
        #Changing feature-values to keys in dictionary that match with input.
        features[elem.lower()][0] = 1

    for key in features.keys():
        #Getting relevant feature and its value
        value, index = features[key][0], features[key][1]

        #Updating values in list to be used for prediction.
        result[index] = value

    #Fetching trained model
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    file_path = os.path.join(data_dir, 'trained_model.sav')
    trainedModel = None
    with open(file_path, 'rb') as f:
        trainedModel = pickle.load(f)

    #Predicting sick leave in weeks
    prediction = trainedModel.predict([result])

    #Returning predicted sick-leave
    return prediction[0]

