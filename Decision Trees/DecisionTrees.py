"""
Español
Autor: German Lopez Rodrigo.
Clase: Machine Learning.
Grupo: 01.
Versión: 1.0
Descripción:
El siguiente programa crea un modelo de arboles de decisión, recibe dos archivos tipo csv
uno para crear el modelo y otro para hacer las pruebas del modelo. En el segúndo archivo
se debe quitar la variable dependiente.

English
Author: German Lopez Rodrigo.
Class: Machine Learning.
Group: 01.
Version: 1.0
Description:
The following program creates a model of decision trees, receives two csv files
one to create the model and another to test the model. In the second file
the dependent variable must be removed.
"""

import math
import argparse
import numpy as np

#----------------------------------------------- Funciones ---------------------------------------------------#
"""
- Objeto para representar los nodos del árbol del modelo de árboles de decisión. (C45)
- Object to represent the tree nodes of the decision tree model. (C45)
"""
class Node(object):
	def __init__(self,root,value,isLeaf):
		self.root = root
		self.values = value
		self.isLeaf = isLeaf
		self.children = []

"""
- Objeto para representar el modelo de árboles de decisión. (C45)
- Object to represent the decision tree model. (C45)
"""
class DecisionTreeC45(object):

	"""
	- Constructor encargado de inicializar los datos necesarios para el algoritmo C45.
	- Constructor in charge of initializing the necessary data for the C45 algorithm.
	"""
	def __init__(self,data,attributes,typeAttributes):
		self.data = data
		self.attributes = attributes
		self.typeAttributes = typeAttributes
		self.classes = self.getClasses(data,-1)
		self.numClasses = len(self.classes)
		self.tree = None

	"""
	- Función encargada de obtener las clases respecto a un atributo de un conjunto de datos.
	- Function in charge of obtaining the classes with respect to an attribute of a data set.
	"""
	def getClasses(self,data,attribute):
		classes = []
		for elem in data:
			if not elem[attribute] in classes:
				classes.append(elem[attribute])
		return classes

	"""
	- Función encargada de obtener y calcular la entropía de un conjunto de datos.
	- Function in charge of obtaining and calculating the entropy of a data set.
	"""
	def getEntropy(self,data):
		sizeData = len(data)

		if sizeData == 0:
			return 0

		classes = [0 for _ in range(self.numClasses)]
		for subset in data:
			classIndex = self.classes.index(subset[-1])
			classes[classIndex] += 1
		classes = [elem/sizeData for elem in classes]

		entropy = 0
		for prob in classes:
			if prob != 0.0:
				entropy -= prob*math.log2(prob)

		return round(entropy,4)

	"""
	- Función encargada de obtener y calcular la ganancia de información de un conjunto de datos.
	- Function in charge of obtaining and calculating the information gain of a data set.
	"""
	def getInformationGain(self,data,subSets):
		#Obteniendo la entropía del atributo dependiente.
		entropyD = self.getEntropy(data)
		sizeData = len(data)
		prob = [round(len(subset)/sizeData,4) for subset in subSets]
		#Obteniendo la entropía del atributo independiente
		entropySubset = 0
		for x in range(0,len(subSets)):
			entropySubset += prob[x]*self.getEntropy(subSets[x])
		return round(entropyD - entropySubset,4)

	"""
	- Función encargada de obtener la ganancia de información y construir los subconjuntos respecto a un atributo.
	- Function in charge of obtaining the information gain and constructing the subsets with respect to an attribute.
	"""
	def getSubSetsAttribute(self,data,attributes):
		subSet = []
		valuesAttributeSubSet = []
		informatioGainMax = -1*float("inf")
		for attribute in attributes:
			indexAttribute = self.attributes.index(attribute)
			if self.typeAttributes[indexAttribute] == "Categorical":
				valuesAttribute = self.getClasses(data,indexAttribute)
				subSets = []

				for elem in valuesAttribute:
					auxSubset = []
					for instances in data:
						if instances[indexAttribute] == elem:
							auxSubset.append(instances)
					subSets.append(auxSubset)
				informatioGain = self.getInformationGain(data,subSets)

				if informatioGain >= informatioGainMax:
					informatioGainMax = informatioGain
					valuesAttributeSubSet = valuesAttribute
					subSet = subSets
					bestAttribute = attribute
			else:
				data.sort(key = lambda elem: elem[indexAttribute])

				auxData = []
				for instances in data:
					if not instances[indexAttribute] in auxData:
						auxData.append(instances[indexAttribute])

				if len(auxData) == 1:
					auxData.append(auxData[0])

				for x in range(0,len(auxData)-1):

					limit = (auxData[x] + auxData[x+1])/2
					lessLimit = []
					greaterLimit = []
					for instances in data:
						if instances[indexAttribute] <= limit:
							lessLimit.append(instances)
						else:
							greaterLimit.append(instances)

					informatioGain = self.getInformationGain(data,[lessLimit,greaterLimit])

					if informatioGain >= informatioGainMax:
						informatioGainMax = informatioGain
						valuesAttributeSubSet = [limit]
						subSet = [lessLimit,greaterLimit]
						bestAttribute = attribute

		return subSet,bestAttribute,valuesAttributeSubSet

	"""
	- Función encargada de evaluar un nuevo conjunto de datos sobre el modelo de árboles de decisión de forma recursiva. (C45)
	- Function responsible for evaluating a new dataset on the decision tree model recursively. (C45)
	"""
	def evaluateTreeRecursive(self,node,data):
		indexAttribute = attributes.index(node.root)

		if self.typeAttributes[indexAttribute] == "Categorical":
			if data[indexAttribute] in node.values:
				indexValue = node.values.index(data[indexAttribute])
				newNode = node.children[indexValue]
				if newNode.isLeaf:
					return newNode.root
				else:
					result = self.evaluateTreeRecursive(newNode,data)
			else:
				return None
		else:

			if data[indexAttribute] <= node.values[0]:
				newNode = node.children[0]
			else:
				newNode = node.children[1]

			if newNode.isLeaf:
				return newNode.root
			else:
				result = self.evaluateTreeRecursive(newNode,data)

		return result

	"""
	- Función encargada de evaluar un nuevo conjunto de datos sobre el modelo de árboles de decisión.
	- Function in charge of evaluating a new data set on the decision tree model.
	"""
	def evaluateDecisionTreeModel(self,data,attributes):
		result = ""
		for x in range(0,len(attributes)-1):
			result +=  str(attributes[x]) + ","
		result += str(attributes[-1]) + "\n"

		for elem in data:
			for y in range(0,len(elem)):
				if type(elem[y]) is str:
					result += str(elem[y]) + ","
				else:
					result += str(round(elem[y],4)) + ","
			result += str(self.evaluateTreeRecursive(self.tree,elem)) + "\n"
		return result

	"""
	- Función encargada de guardar los resultados en un archivo "csv".
	- Function in charge of saving the results in a "csv" file.
	"""
	def saveResults(self,data,attributes):
		file = open("./Predicciones.csv", "w")
		file.write(self.evaluateDecisionTreeModel(data,attributes))
		file.close()

	"""
	- Función encargada de guardar el modelo de árboles de decisión en un archivo json.
	- Function in charge of saving the decision tree model in a json file.
	"""
	def saveDecisionTree(self):
		file = open("./DecisionTree.json", "w")
		tree = ""
		tree += self.buildViewDecisionTree(self.tree,"")[:-2]
		file.write(tree[1:])
		file.close()

	"""
	- Función encargada de transformar el modelo de árboles de decisión en un archivo json.
	- Function in charge of transforming the decision tree model into a json file.
	"""
	def buildViewDecisionTree(self,node,indent=""):
		tree = ""
		if not node.isLeaf:
			indexAttribute = self.attributes.index(node.root)
			if self.typeAttributes[indexAttribute] == "Categorical":
				tree += "\n" + indent + "{\n" + indent + "\"Root\": \"" + str(node.root) + "\",\n"
				tree += indent + "\"Decision\": \n" + indent + "\t\t{\n"
				for x in range(0,len(node.values)):
					tree += indent + "\t\t\"" + str(node.values[x]) + "\": "
					tree += self.buildViewDecisionTree(node.children[x], indent + "\t\t\t")

					if x == len(node.values)-1:
						tree = tree[:-2] + "\n"

				tree += indent + "\t\t}\n" + indent + "},\n"

			else:
				tree += "\n" + indent + "{\n" + indent + "\"Root\": \"" + str(node.root) + "\",\n"
				tree += indent + "\"Decision\": \n" + indent + "\t\t{\n"

				leftChild = node.children[0]
				rightChild = node.children[1]

				#Caso cuando es menor o igual(<=)
				if leftChild.isLeaf:
					tree += indent + "\t\t\"<= " + str(node.values[0]) + "\":"
					tree += " \"" + str(leftChild.root) + "\",\n"
				else:
					tree += indent + "\t\t\"<= " + str(node.values[0]) + "\":"
					tree += self.buildViewDecisionTree(leftChild, indent + "\t\t\t")


				#Caso cuando es mayor (>).
				if rightChild.isLeaf:
					tree += indent + "\t\t\"> " + str(node.values[0]) + "\":"
					tree += " \"" + str(rightChild.root) + "\"\n"
				else:
					tree += indent + "\t\t\"> " + str(node.values[0]) + "\":"
					tree += self.buildViewDecisionTree(rightChild, indent + "\t\t\t")[:-2] + "\n"

				tree += indent + "\t\t}\n" + indent + "},\n"
		else:
			tree +=  "\"" + str(node.root) + "\",\n"

		return tree

	"""
	- Función encargada de validar si los datos son de la misma clase.
	- Function in charge of validating if the data is of the same class.
	"""
	def getSameClass(self,data):
		for instances in data:
			if instances[-1] != data[0][-1]:
				return False
		return True

	"""
	- Función encargada de obtener la clase más común de un conjunto de datos.
	- Function in charge of obtaining the most common class of a data set.
	"""
	def getMostCommonClass(self,data):
		freq = [0]*len(self.classes)
		for instances in data:
			index = self.classes.index(instances[-1])
			freq[index] += 1
		classId = freq.index(max(freq))
		return self.classes[classId]

	"""
	- Función encargada de generar el modelo de árboles de decisión.
	- Function in charge of generating the decision tree model.
	"""
	def generateTree(self):
		self.tree = self.generateTreeRecursive(self.data,self.attributes)

	"""
	- Función encargada de generar el modelo de árboles de decisión de forma recursiva.
	- Function in charge of generating the decision tree model recursively.
	"""
	def generateTreeRecursive(self,data,attributes):
		if len(data) == 0:
			return Node("Fail",[],True)
			
		elif self.getSameClass(data):
			return Node(data[0][-1],[],True)

		elif len(attributes) == 0:
			return Node(self.getMostCommonClass(data),[],True)

		else:
			subSets,bestAttribute,valuesAttribute = self.getSubSetsAttribute(data,attributes)
			newAttributes = attributes[:]
			newAttributes.remove(bestAttribute)
			node = Node(bestAttribute,valuesAttribute,False)
			node.children = [self.generateTreeRecursive(subset,newAttributes) for subset in subSets]
			return node

"""
- Función encargada de cargar los datos almacenados en un archivo "csv" a memoria.
- Function in charge of loading the data stored in a "csv" file into memory.
"""
def readFileCsv(file):
	file = open(file)
	attributes = file.readline().strip().split(",")
	numColumns = len(attributes)
	typeAttributes = []
	data = []
	for line in file:
		auxData = line.strip().split(",")
		transformedData = []
		typeDate = ""
		for elem in auxData:
			auxElem = elem.isdigit()
			if auxElem or "." in elem or "-" in elem:
				typeDate = "number"
				transformedData.append(float(elem))
			else:
				transformedData.append(elem)
		data.append(transformedData)

	file.close()

	for x in range(0,numColumns):
		flag = True
		auxType = type(data[0][x])
		for instances in data:
			if auxType != type(instances[x]):
				flag = False
				break

		if flag:
			if auxType is str:
				typeAttributes.append("Categorical")
			else:
				typeAttributes.append("Numeric")
		else:
			print("\nError in struct of the data\n")
			exit()

	return data,attributes,typeAttributes

#-----------------------------------------Preparación de los Datos--------------------------------------------------#

parser = argparse.ArgumentParser(description='Decision Trees')
parser.add_argument('inDirTraining', type=str, help='Input directory for Training Data')
parser.add_argument('inDirTest', type=str, help='Input directory for test Data')
args = parser.parse_args()
#Leyendo archivo de la fuente de datos para la construcción del modelo de regresión linear.
data,attributes,typeAttributes = readFileCsv(args.inDirTraining + ".csv")
#Construllendo el modelo de árboles de decisión.
arbol = DecisionTreeC45(data,attributes[:-1],typeAttributes)
arbol.generateTree()
arbol.saveDecisionTree()
#Evaluación del modelo de árboles de decisión.
data,_,typeAttributes = readFileCsv(args.inDirTest + ".csv")
arbol.saveResults(data,attributes)

