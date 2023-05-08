from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from ai_model import readAndProcessData, trainModel, predict

def main():

    """
        Uncomment the model you want to train.
    """
    #model = MLPClassifier(activation="relu", solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(32, 16, 3), random_state=1) # https://scikit-learn.org/stable/modules/neural_networks_supervised.html
    #model = GaussianNB()

    """
        Uncomment to fetch data, process data and train model.
        If no changes has been made, this only has to be executed once.
    """
    training_X, training_Y, test_X, test_Y = readAndProcessData()
    #trainModel(training_X, training_Y, test_X, test_Y, model)

    
    """
        Uncomment to predict sick-leave in weeks for a persona with feature values specified below.
        Variables in 'features' are changable, see predict() in ai_model.py to see what other values can be used.
    """
    #features = ['troms og finnmark', '16-19', 'diseases in the digestive organs', 'female']
    #result = predict(features)
    #print(f"Result: {result} weeks") 


#Automatically calls main-function when file is executed
if __name__ == "__main__":
    main()
