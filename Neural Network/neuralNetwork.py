"""
Español
Autor: German Lopez Rodrigo.
Clase: Machine Learning.
Grupo: 01.
Versión: 1.0
Descripción:
El siguiente programa implementa una red neuronal.

English
Author: German Lopez Rodrigo.
Class: Machine Learning.
Group: 01.
Version: 1.0
Description:
The following program implements a neural network.
"""
import argparse
import random
import numpy as np

#----------------------------------------------- Funciones ---------------------------------------------------#

"""
- Objeto para representar las neuronas de la red neuronal.
- Object to represent the neurons of the neural network.
"""
class Neuron(object):
	"""	
	- Constuctor encargado de crear una neurona con pesos aleatroios y con n de salidas (Axon).
	- Constuctor in charge of creating a neuron with random weights and n outputs (Axon).
	"""
	def __init__(self,idNeuron,weights,numAxon):
		self.idNeuron = idNeuron
		self.weights = weights
		self.numAxon = numAxon
		self.outputs = []

	"""	
	- Función encargada de activar la neurona mediante la multiplicación de los pesos por sus entradas.
	- Function responsible for activating the neuron by multiplying the weights by their inputs.
	"""
	def activationNeuron(self,data):
		self.outputs = []
		for _ in range(0,self.numAxon):
			ini = 0 
			auxData = [1] + data
			for x in range(0,len(auxData)):
				ini += auxData[x]*self.weights[x]
			outReal = self.sigmoideFunction(ini)
			self.outputs.append(outReal)

	"""	
	- Función encargada de activar la neurona de entrada.
	- Function responsible for activating the neuron by multiplying the weights by their inputs.
	"""
	def activationNeuronInput(self,data):
		self.outputs = []
		for _ in range(0,self.numAxon):
			self.outputs.append(data)


	"""
	- Función encargada de realizar la normalización de un dato utilizando la función de activación sigmoide.
	- Function responsible of normalizing a data using the sigmoide activation function.
	"""
	def sigmoideFunction(self,x):
		sigmoide = 1/(1+np.exp(-x))
		return round(sigmoide,6)

	def __repr__(self):
		result = ""
		for elemt in self.weights:
			result +=  str(elemt) + ","
		result = "{\"Id\": \"" + self.idNeuron + "\",\"Weights\": (" + result[:-1] + "), \"Axon\": " + str(self.numAxon) + "}"
		return result
	
	def __str__(self):
		result = ""
		for elemt in self.weights:
			result +=  str(elemt) + ","
		result = "{\"Id\": \"" + self.idNeuron + "\",\"Weights\": (" + result[:-1] + "), \"Axon\": " + str(self.numAxon) + "}"
		return result

"""
- Función encargada de crear la estructura de la red neuronal.
- Function responsible of creating the structure of the neural network.
"""
def buildNeuralNetwork(numHiddenLayers,numInputLayerNeurons,numHiddenLayerNeurons,numOutputLayerNeurons,learningFactor,numInteractions):
	neuralNetwork = {"Input":[],"Hidden": [],"Output": [], "numHiddenLayers": numHiddenLayers, "numInputLayerNeurons": numInputLayerNeurons, 
	"numHiddenLayerNeurons": numHiddenLayerNeurons,  "numOutputLayerNeurons": numOutputLayerNeurons,
	 "learningFactor": learningFactor,"numInteractions": numInteractions}
	
	#Capa de Entrada.
	inputNeurons = []
	for count in range(0,numInputLayerNeurons):
		idNeuron = "Input Neuron " + str(count+1)
		weights = [1 for _ in range(0,numHiddenLayerNeurons)]
		inputNeurons.append(Neuron(idNeuron,weights,numHiddenLayerNeurons))
	neuralNetwork["Input"] = inputNeurons

	#Capas Ocultas.
	hiddenLayersNeurons = []
	for index in range(0,numHiddenLayers):
		hiddenNeurons = []
		for count in range(0,numHiddenLayerNeurons):
			idNeuron = "Hidden Neuron " + str(index+1) + "," + str(count+1)
			numWeights = numInputLayerNeurons if index == 0 else numHiddenLayerNeurons
			weights	 = [round(random.uniform(-1,1),4) for _ in range(0,numWeights+1)]
			numNeruons = numOutputLayerNeurons if index == numHiddenLayers-1 else numHiddenLayerNeurons
			hiddenNeurons.append(Neuron(idNeuron,weights,numNeruons))
		hiddenLayersNeurons.append(hiddenNeurons)
	neuralNetwork["Hidden"] = hiddenLayersNeurons

	#Capa de salida.
	outputNeurons = []
	for count in range(0,numOutputLayerNeurons):
		idNeuron = "Output Neuron " + str(count+1)
		weights = [round(random.uniform(-1,1),4) for _ in range(0,numHiddenLayerNeurons+1)]
		outputNeurons.append(Neuron(idNeuron,weights,1))
	neuralNetwork["Output"] = outputNeurons

	return neuralNetwork

"""
- Función encargada de insertar en la red neuronal un dato de entrada.
- Function responsible for inserting an input data into the neural network.
"""
def evaluateNeuralNetwork(data,neuralNetwork):
	numHiddenLayers = neuralNetwork["numHiddenLayers"]
	numInputLayerNeurons = neuralNetwork["numInputLayerNeurons"]
	numHiddenLayerNeurons = neuralNetwork["numHiddenLayerNeurons"]
	numOutputLayerNeurons = neuralNetwork["numOutputLayerNeurons"]

	#Para cada neurona 'xIn' perteneciente a la capa de entrada.
	outputsInputLayer = []
	for xIn in range(0,numInputLayerNeurons):
		neuronxInput = neuralNetwork["Input"][xIn]
		neuronxInput.activationNeuronInput(data[xIn])
		outputsInputLayer.append(neuronxInput.outputs)

	#Para cada capa oculta 'xHidden' de la red neuronal.
	outputsHiddenLayers = [outputsInputLayer]
	for xHidden in range(0,numHiddenLayers):
		outputsHiddenLayer = []
		#Para cada neurona 'yHidden' perteneciente a la capa oculta "xHidden".
		for yHidden in range(0,numHiddenLayerNeurons):
			neuronxHidden = neuralNetwork["Hidden"][xHidden][yHidden]
			neuronxData = [outputsHiddenLayers[(xHidden+1)-1][z][yHidden] for z in range(0,len(outputsHiddenLayers[(xHidden+1)-1]))]
			neuronxHidden.activationNeuron(neuronxData)
			outputsHiddenLayer.append(neuronxHidden.outputs)
		outputsHiddenLayers.append(outputsHiddenLayer)
	
	#Para cada neurona 'yOut' perteneciente a la capa de salida.
	outputsOutputLayer = []
	for yOut in range(0,numOutputLayerNeurons):
		neuronxOutput = neuralNetwork["Output"][yOut]
		outputData = [outputsHiddenLayers[-1][z][yOut] for z in range(0,len(outputsHiddenLayers[-1]))]
		neuronxOutput.activationNeuron(outputData)
		outputsOutputLayer.append(neuronxOutput.outputs)

	return outputsHiddenLayers,outputsOutputLayer

"""
- Función encargada de realizar el ajuste de pesos por retropropagacion del error.
- Function responsibleof adjusting weights by backpropagating the error.
"""
def backpropagationAlgorithm(data,neuralNetwork,outputsHiddenLayers,outputsOutputLayer):
	numHiddenLayers = neuralNetwork["numHiddenLayers"]
	numHiddenLayerNeurons = neuralNetwork["numHiddenLayerNeurons"]
	numOutputLayerNeurons = neuralNetwork["numOutputLayerNeurons"]
	learningFactor = neuralNetwork["learningFactor"]

	errorsNeuron = []
	oldWeights = []
	#Para cada neurona 'yOut' perteneciente a la capa de salida se realiza el ajuste de pesos por retropropagacion del error.
	for yOut in range(0,numOutputLayerNeurons):
		neuron = neuralNetwork["Output"][yOut]
		outputNeuron = neuron.outputs[0]
		derivative = derivativeSigmoideFunction(outputNeuron)
		errorNeuron = (outputNeuron-data[yOut][-1])*derivative
		inputsNeuron = [1] + [outputsHiddenLayers[-1][z][0] for z in range(0,len(outputsHiddenLayers[-1]))]
		oldWeights.append(neuron.weights)
		errorsNeuron.append(errorNeuron)
		#Calculo de nuevos pesos para la capa de salida.
		if errorNeuron != 0:
			newWeights = [round((neuron.weights[z]-learningFactor*inputsNeuron[z]*errorNeuron),6) for z in range(0,len(neuron.weights))]
			neuralNetwork["Output"][yOut].weights = newWeights

	#Calculo de nuevos pesos para cada neurona 'yHidden' perteneciente a la capa oculta "xHidden".
	for xHidden in range(numHiddenLayers-1,-1,-1):
		
		errorsNeuron2 = []
		oldWeights2 = []

		for yHidden in range(0,numHiddenLayerNeurons):
			hiddenNeuron = neuralNetwork["Hidden"][xHidden][yHidden] # nodo j 
			outputHiddenNeuron = hiddenNeuron.outputs[0] #aj
			derivative = derivativeSigmoideFunction(outputHiddenNeuron)

			#∆j
			summation = 0
			for index in range(0,len(hiddenNeuron.outputs)):
				summation += oldWeights[index][yHidden+1]*errorsNeuron[index]
			
			errorHiddenNeuron = derivative*summation 
			inputsHiddenNeuron = [1] + [outputsHiddenLayers[xHidden][z][yHidden] for z in range(0,len(outputsHiddenLayers[xHidden]))]
			oldWeights2.append(hiddenNeuron.weights)
			errorsNeuron2.append(errorHiddenNeuron)

			#Calculo de nuevos pesos para la capa oculta "xHidden".
			if errorHiddenNeuron != 0:
				newWeights = [round(hiddenNeuron.weights[z]-learningFactor*inputsHiddenNeuron[z]*errorHiddenNeuron,6) for z in range(0,len(hiddenNeuron.weights))]
				neuralNetwork["Hidden"][xHidden][yHidden].weights = newWeights

		errorsNeuron = errorsNeuron2
		oldWeights = oldWeights2

	return neuralNetwork

"""
- Función encargada de entrenar la red neuronal para obtener los mejores pesos.
- Function responsible of training the neural network to obtain the best weights.
"""
def trainingNeuralNetwork(data,neuralNetwork):
	numHiddenLayers = neuralNetwork["numHiddenLayers"]
	numInputLayerNeurons = neuralNetwork["numInputLayerNeurons"]
	numHiddenLayerNeurons = neuralNetwork["numHiddenLayerNeurons"]
	numOutputLayerNeurons = neuralNetwork["numOutputLayerNeurons"]
	numInteractions = neuralNetwork["numInteractions"]
	#Contador del número de Interacciones.
	auxNumInteractions = 0
	#Repetir hasta que se cumpla la condición de terminación.
	while(True):
		#Error cuadratico general.
		#errorTotal = 0.0
		#Saltos en el conjunto de entrenamiento según las neuronas de salida.
		jump = 1 if numOutputLayerNeurons == 1 else numOutputLayerNeurons
		#Para cada (⃗x,y) del conjunto de entrenamiento.
		for i in range(0,len(data),jump):
			#Evaluando la red neuronal con un dato de entrada.
			outputsHiddenLayers,outputsOutputLayer = evaluateNeuralNetwork(data[i][:-1],neuralNetwork)
			#Datos de salida reales.
			auxData = [data[i] if numOutputLayerNeurons == 1 else data[i+z] for z in range(0,numOutputLayerNeurons)]

			#La salida de la red de cada neurona se compara con la salida deseada para calcular el error en cada unidad:
			#error = 0 
			#for neuron in range(0,numOutputLayerNeurons):
			#	error += (outputsOutputLayer[neuron][0] - data[neuron][-1])**2
			#quadraticError = (1/2)*(error)

			#Ajuste de pesos por retropropagacion del error.
			neuralNetwork = backpropagationAlgorithm(auxData,neuralNetwork,outputsHiddenLayers,outputsOutputLayer)

			#errorTotal += (quadraticError)

		#La condición de terminación es que se llegue al número de interacciones indicadas.
		auxNumInteractions += 1
		if auxNumInteractions > numInteractions:
			break

	return neuralNetwork

"""
- Función encargada de realizar la normalización de un dato utilizando la función de activación sigmoide.
- Function responsible of normalizing a data using the sigmoide activation function.
"""
def sigmoideFunction(x):
	if x < 0.5:
		return 0
	return 1

"""
- Función encargada de evaluar la red neuronal.
- Function in charge of evaluating the neural network.
"""
def testNeuralNetwork(data,neuralNetwork):
	numOutputLayerNeurons = neuralNetwork["numOutputLayerNeurons"]
	results = []
	jump = 1 if numOutputLayerNeurons == 1 else numOutputLayerNeurons
	for i in range(0,len(data),jump):
		auxData = [data[i] if numOutputLayerNeurons == 1 else data[i+z] for z in range(0,numOutputLayerNeurons)]
		_,result = evaluateNeuralNetwork(data[i],neuralNetwork)
		for elem in result:
			results.append(sigmoideFunction(elem[0]))
	print(results)
	return results


def derivativeSigmoideFunction(x):
	return x*(1-x)

"""
- Función encargada de guardar la red neuronal con sus pesos en un archivo json.
- Function in charge of saving the classification of the data in a csv file.
"""
def saveFileJson(neuralNetwork,file):
	file = open(file+"NeuralNetwork.json", "w")
	RN = "{\n"
	for key in neuralNetwork.keys():
		aux = str(neuralNetwork.get(key))
		aux = aux.replace('(','[').replace(')',']')
		RN += "\t\"" + str(key)  + "\": " + aux + ",\n"
	RN = RN[:-2] + "\n"
	file.write(RN+"}\n")
	file.close()


"""
- Función encargada de guardar la clasificación de los datos en un archivo csv.
- Function in charge of saving the classification of the data in a csv file.
"""
def saveFileCsv(data,attributes,file,results):
	file = open(file+"Results.csv", "w")

	result = ""
	for elem in attributes:
		result += elem + ","
	result = result[:-1] + "\n"

	for indexY in range(0,len(data)):
		for indexX in range(0,len(attributes)-1):
			result += str(data[indexY][indexX]) + ","
		result = result[:-1] + "," + str(results[indexY]) + "\n"
	file.write(result+"\n")
	file.close()


"""
- Función encargada de cargar los datos almacenados en un archivo "csv" a memoria.
- Function responsible of loading the data stored in a "csv" file into memory.
"""
def readFileCsv(file):
	file = open(file)
	attributes = file.readline().strip().split(",")
	data = []
	for line in file:
		auxData = line.strip().split(",")
		transformedData = []
		for elem in auxData:
			auxElem = elem.isdigit()
			if auxElem or "." in elem or "-" in elem:
				transformedData.append(float(elem))
			else:
				transformedData.append(elem)
		data.append(transformedData)
	file.close()
	return data,attributes


#-----------------------------------------Preparación de los Datos--------------------------------------------------#
parser = argparse.ArgumentParser(description='Neural Network', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('inDirTraining', type=str, help='Input directory for Training Data')
parser.add_argument('inDirTest', type=str, help='Input directory for test Data')
parser.add_argument('learningFactor', type=float, help='Learning Factor.\n')
parser.add_argument('numHiddenLayerNeurons', type=int, help='Number of neurons in each hidden layer.\n')
parser.add_argument('numOutputLayerNeurons', type=int, help='Number of neurons in the output layer..\n')
parser.add_argument('numHiddenLayers', type=int, help='Number of hidden layers..\n')
parser.add_argument('numInteractions', type=int, help='Number of interactions.\n')
args = parser.parse_args()

#-------------------------------------- Construcción de la Red Neuronal --------------------------------------------#
#Leyendo archivo de la fuente de datos para la construcción del modelo de regresión linear.
data,attributes = readFileCsv(args.inDirTraining + ".csv")
#Número de Variables de entrada. 
numVar = len(attributes)-1
#Factor de aprendizaje
learningFactor = args.learningFactor
#Número de variables en la capa de entrada.
numInputLayerNeurons = numVar
#Número de nueronas en las capas ocultas.
numHiddenLayerNeurons = args.numHiddenLayerNeurons
#Número de nueronas en la capa de salida.
numOutputLayerNeurons = args.numOutputLayerNeurons
#Número de capas ocultas.
numHiddenLayers = args.numHiddenLayers
#Número de interacciones.
numInteractions = args.numInteractions
#Red Neuronal.
neuralNetwork = buildNeuralNetwork(numHiddenLayers,numInputLayerNeurons,numHiddenLayerNeurons,numOutputLayerNeurons,learningFactor,numInteractions)
#Entrenando Red Neuronal.
neuralNetwork = trainingNeuralNetwork(data,neuralNetwork)
#Almacenando Red Neuronal.
saveFileJson(neuralNetwork,args.inDirTraining)

#-------------------------------------- Evaluación de la Red Neuronal --------------------------------------------#
dataTest,_ = readFileCsv(args.inDirTest + ".csv")
#Evaluando la red neuronal.
results = testNeuralNetwork(dataTest, neuralNetwork)
#Almacenando los resultados en un archivo csv.
saveFileCsv(dataTest,attributes,args.inDirTraining,results)


