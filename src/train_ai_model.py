from sklearn.neural_network import MLPClassifier
from ai_model import readAndProcessData, trainModel, getResult

#Resurs til å konvertere kategoriske features fra string til int.
#https://www.projectpro.io/recipes/convert-string-categorical-variables-into-numerical-variables-using-label-encoder

training_X, training_Y, test_X, test_Y = readAndProcessData()
model = MLPClassifier(activation="relu", solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(32, 16, 3), random_state=1) # https://scikit-learn.org/stable/modules/neural_networks_supervised.html
#model = GaussianNB()
#model = Perceptron(tol=1e-5, random_state=0)

trainedModel = trainModel(training_X, training_Y, test_X, test_Y, model)
result = getResult(['Agder', '16-19', 'pregnancy disorders', 'FEMALE'])
print("!!!!", result)
