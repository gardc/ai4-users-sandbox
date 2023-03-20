

#from sklearn.naive_bayes import GaussianNB
#from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
#from sklearn.linear_model import Perceptron
import numpy as np
import pandas as pd
import os



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
 

#Combining data into single user datas, and setting meanValue of attributeValues as labels
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
  #  print(tempDf[:10])
   # for col in tempDf.columns:
    #    print("Kolonne: ", col)
#Splitting into training and test-data. (80/20)   
    trainDf, testDf = train_test_split(tempDf, test_size=0.2, random_state=42)

#Converting from DataFrame to array, and seperating label and features. X contains features, and Y contains corresponding labels.
    training_X = trainDf.to_numpy().tolist()
    training_Y = [elem.pop(0) for elem in training_X]

    test_X = testDf.to_numpy().tolist()
    test_Y = [elem.pop(0) for elem in test_X]


#Returning generated datasets. 
    return training_X, training_Y, test_X, test_Y


def trainModel(training_X, training_Y, test_X, test_Y, model):
    trainedModel = model.fit(training_X, training_Y)
    predicted = trainedModel.predict(test_X)
    correct = 0
    for i in range(len(predicted)):
        if predicted[i] == test_Y[i]:
            correct += 1
       # print(f"result: {predicted[i]}:{test_Y[i]}")
    print(f"Accuracy: ", correct/len(predicted))
    return trainedModel


def getResult(input):
#Converting input from string to int
    result = np.zeros((1, 31), dtype=int).tolist()[0]
    features = {'Agder': [0, 0], 'Innlandet': [0, 1], 'Møre og Romsdal': [0, 2], 'Nordland': [0, 3], 'Oslo': [0, 4], 'Rogaland': [0, 5],
                'Troms og Finnmark': [0, 6], 'Trøndelag': [0, 7], 'Vestfold og Telemark': [0, 8], 'Vestland': [0, 9], 'Viken': [0, 10],
                '16-19': [0, 11], '20-24': [0, 12], '25-29': [0, 13], '30-34': [0, 14], '35-39': [0, 15], '40-44': [0, 16],
                '45-49': [0, 17], '50-54': [0, 18], '55-59': [0, 19], '60-64': [0, 20], '65-69': [0, 21], 'cardiovascular diseases': [0, 22],
                'muscle/skeleton disorders': [0, 23], 'mental disorders': [0, 24], 'pregnancy disorders': [0, 25], 'disease in the digestive organs': [0, 26],
                'diseases in the respiratory tract': [0, 27], 'diseases in the nervous system': [0, 28], 'man': [0, 29], 'woman': [0, 30]}

    for elem in input:
    #Changing feature-values for keys in dictionary that match with input.
        features[elem][0] = 1

    for key in features.keys():
    #Getting relevant feature and its value
        value, index = features[key][0], features[key][1]
    #Updating values in list to be used for prediction.
        result[index] = value

#Predicting sick leave in weeks
    prediction = trainedModel.predict([result])

#Returning predicted sick-leave
    return prediction[0]
    


#Resurs til å konvertere kategoriske features fra string til int.
#https://www.projectpro.io/recipes/convert-string-categorical-variables-into-numerical-variables-using-label-encoder

training_X, training_Y, test_X, test_Y = readAndProcessData()
model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1) # https://scikit-learn.org/stable/modules/neural_networks_supervised.html
#model = GaussianNB()
#model = Perceptron(tol=1e-5, random_state=0)

trainedModel = trainModel(training_X, training_Y, test_X, test_Y, model)
result = getResult(['Agder', '16-19', 'pregnancy disorders', 'woman'])
#print(result)


