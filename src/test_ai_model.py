from . import ai_model

#Resurs til å konvertere kategoriske features fra string til int.
#https://www.projectpro.io/recipes/convert-string-categorical-variables-into-numerical-variables-using-label-encoder

training_X, training_Y, test_X, test_Y = ai_model.readAndProcessData()
model = ai_model.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1) # https://scikit-learn.org/stable/modules/neural_networks_supervised.html
#model = GaussianNB()
#model = Perceptron(tol=1e-5, random_state=0)

trainedModel = ai_model.trainModel(training_X, training_Y, test_X, test_Y, model)
result = ai_model.getResult(['Agder', '16-19', 'pregnancy disorders', 'woman'])
#print(result)
