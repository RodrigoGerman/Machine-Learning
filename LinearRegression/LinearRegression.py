"""
Español
Autor: German Lopez Rodrigo.
Clase: Machine Learning.
Grupo: 01.
Versión: 1.0
Descripción:
El siguiente programa crea un modelo de regresión linear, recibe dos archivos tipo csv
uno para crear el modelo y otro para hacer las pruebas del modelo. En el segúndo archivo
se debe quitar la variable dependiente.

English
Author: German Lopez Rodrigo.
Class: Machine Learning.
Group: 01.
Version: 1.0
Description:
The following program creates a linear regression model, receives two csv files
one to create the model and another to test the model. In the second file
the dependent variable must be removed.
"""

import math
import numpy as np
import argparse

#----------------------------------------------- Funciones ---------------------------------------------------#
"""
- Objeto para representar los coeficientes del modelo de regresión lineal. (betas)
- Object to represent the coefficients of the linear regression model. (betas)
"""
class Variable(object):
	def __init__(self,coefficient,variables,grade):
		self.coefficient = coefficient
		self.variable = variables
		self.grade = grade

	def __repr__(self):
		result = str(self.coefficient) 
		for b in self.variable:
			result += "B"+str(b)+"ˆ" + str(self.grade)
		return result
	
	def __str__(self):
		result = str(self.coefficient) 
		for b in self.variable:
			result += "B"+str(b)+"ˆ" + str(self.grade)
		return result

"""
- Función encargada de crear los polinomios factorizados de la forma (a,bBo,cB1,....,nBn) de un set de datos.
- Function in charge of creating the factored polynomials of the form (a,bBo,cB1,....,nBn) of a data set.
"""
def createPolynomials(data,sizeData,numVariables):
	polynomials = []
	for x in range(0,sizeData):
		terminus = [transformedData[-1][x],Variable(-1,[0],1)]
		for y in range(1,numVariables):
			terminus.append(Variable(-transformedData[y-1][x],[y],1))
		polynomials.append(terminus)
	return polynomials

"""
- Función encargada de calcular el cuadrado de un polinomio.
- Function in charge of calculating the square of a polynomial.
"""
def squaredPolynomial(polynomial):
	polynomialResult = []
	for i in range(0,len(polynomial)):
		#Si es un coeficiente.
		if type(polynomial[i]) is Variable:
			polynomialResult.append(Variable(polynomial[i].coefficient*polynomial[i].coefficient,polynomial[i].variable,2))
			for term in polynomial[i+1:]:
				if type(term) is Variable:
					polynomialResult.append(Variable(term.coefficient*polynomial[i].coefficient*2,polynomial[i].variable + term.variable ,1))
		#Si es el termino independiente.
		else:
			polynomialResult.append(polynomial[i]*polynomial[i])
			for term in polynomial[i+1:]:
				if type(term) is Variable:
					polynomialResult.append(Variable(term.coefficient*polynomial[i]*2,term.variable,1))
	return polynomialResult

"""
- Función encargada de calcular la derivada de un polinomio respecto a una variable.
- Function in charge of calculating the derivative of a polynomial with respect to a variable.
"""
def derivedPolynomial(polynomial,variable):
	derived = []
	for term in polynomial:
		auxType =  type(term)
		if auxType is Variable:
			#Si es un valor con la variable a derivar.
			if variable in term.variable:
				#Coeficientes con sólo la variable a derivar.
				if term.grade == 1 and len(term.variable) == 1:
					derived.append(term.coefficient)
				#Coeficientes con más variables.
				elif term.grade == 1 and len(term.variable) > 1:
					auxVariables = []
					for aux in term.variable:
						if variable != aux:
							auxVariables.append(aux)
					derived.append(Variable(term.coefficient,auxVariables,1))
				#Coeficientes al cuadrado.
				elif term.grade == 2:
					derived.append(Variable(term.coefficient*2,term.variable,1))
	return derived

"""
- Función encargada de construir y regresar un polinomio.
- Function in charge of building and returning a polynomial.
"""
def getPolinomial(polinomial,name):
	polynomial = "\"" + name + "\": \""
	for x in range(0,len(polinomial)):
		if type(polinomial[x]) is Variable:
			if polinomial[x].coefficient < 0:
				polynomial += str(polinomial[x])
			else:
				polynomial += "+" + str(polinomial[x])
		else:
			if polinomial[x] < 0:
				polynomial += str(polinomial[x])
			else:
				polynomial += "+" + str(polinomial[x])
	return polynomial + "\""

"""
- Función encargada de crear el polinomio S de un set de datos.
- Function in charge of creating the polynomial S of a data set.
"""
def buildPolynomialS(developedPolynomials):
	polynomialS = []
	numPolynomials = len(developedPolynomials)
	numTerm = len(developedPolynomials[0])
	#Suma de los polinomios.
	for x in range(0,numTerm):
		sumValues = 0	
		for y in range(0, numPolynomials):
			auxType = type(developedPolynomials[y][x])
			#Sumando los valores de los coeficientes.
			if auxType is Variable:
				sumValues += developedPolynomials[y][x].coefficient
			#Sumando los terminos independientes.
			else:
				sumValues += developedPolynomials[y][x]

		if auxType is Variable:
			polynomialS.append(Variable(round(sumValues,8),developedPolynomials[y][x].variable,developedPolynomials[y][x].grade))
		else:
			polynomialS.append(round(sumValues,8))

	return polynomialS

"""
- Función encargada de normalizar una ecuación sobre el valor recibido.
- Function in charge ofr normalizing an equation on the received value.
"""
def normalizeEquation(equation,value):
	auxEquation = []
	pivot = equation[value+1].coefficient

	if pivot == 0:
		return equation

	for term in equation:
		if type(term) is Variable: 
			term.coefficient = term.coefficient/pivot
			auxEquation.append(term)
		else:
			auxEquation.append(term/pivot)

	return auxEquation

"""
- Función encargada de realizar la resta entre una ecuación base sobre otras ecuaciones.
- Function in charge of subtracting a base equation from other equations.
"""
def subtractionEquations(equation,equations,pivot):
	equationsResult = [equation]
	for x in range(0,len(equations)):
		auxEquation = []
		if equations[x][pivot+1].coefficient != 0.0:
			for y in range(0,len(equation)):
				if type(equations[x][y]) is Variable:
					if equations[x][y].coefficient != 0.0:
						equations[x][y].coefficient = equations[x][y].coefficient - equation[y].coefficient
					auxEquation.append(equations[x][y])
				else:
					if equations[x][y] != 0.0:
						auxEquation.append(equations[x][y] - equation[y])
					else:
						auxEquation.append(equations[x][y])
			equationsResult.append(auxEquation)
		else:
			equationsResult.append(equations[x])
	return equationsResult

"""
- Función encargada de calcular y resolver el sistema de ecuaciones.
- Function in charge of calculating and solving the system of equations.
"""
def solvingSystemEquations(equations,numVariables):
	for x in range(0, numVariables):
		auxEquation = []
		for equation in equations[x:]:
			auxEquation.append(normalizeEquation(equation,x))
		equations = equations[:x] + auxEquation
		equations = equations[:x] + subtractionEquations(equations[x],equations[x+1:],x)
	return equations

"""
- Función encargada de construir y regresar el sistema de ecuaciones.
- Function in charge of building and returning the system of equations.
"""
def getSystemEquations(equations,name):
	system = "\""+name+"\":\n \t\t{"
	for x in range(0,len(equations)):
		system+= "\n\t\t\t" + getPolinomial(equations[x],str(x)) + ","
	return system[:-1] + "\n\t\t}" 

"""
- Función encargada de calcular los coeficientes del modelo de regresión lineal. (betas)
- Function in charge of calculating the coefficients of the linear regression model. (betas)
"""
def calculatingBetas(equations,numVariables):
	betas = [round(equations[-1][0]*-1,8)]
	for x in range(numVariables-2,-1,-1):
		auxBetas = 0
		auxValue = equations[x][0]*-1
		for y in range(numVariables,x+1,-1):
			if type(equations[x][y]) is Variable:
				auxValue += equations[x][y].coefficient*betas[auxBetas]*-1
			auxBetas += 1
		betas.append(round(auxValue,8))
	return betas

"""
- Función encargada de construir y regresar el modelo de regresión lineal.
- Function responsible for building and returning the linear regression model.
"""
def getLinearRegressionModel(linearRegressionModel,flagLogistic):
	model = "\"Ecuación\": "
	if not flagLogistic:
		model += "\"y = " + str(linearRegressionModel[0])
		for x in range(1,len(linearRegressionModel)):
			if linearRegressionModel[x] < 0:
				model += str(linearRegressionModel[x]) + "x" + str(x-1)
			else:
				model += "+" + str(linearRegressionModel[x]) + "x" + str(x-1)

	else:
		model += "\"y = 1/(1+e^-(" + str(linearRegressionModel[0])
		for x in range(1,len(linearRegressionModel)):
			if linearRegressionModel[x] < 0:
				model += str(linearRegressionModel[x]) + "x" + str(x-1)
			else:
				model += "+" + str(linearRegressionModel[x]) + "x" + str(x-1)
		model += "))"

	return model + "\""
"""
- Función encargada de evaluar el modelo de regresión lineal.
- Function in charge of evaluating the linear regression model.
"""
def evaluateLinearRegressionModel(linearRegressionModel,data,categoricalOutputLabels,flagLogistic):
	numVariables = len(data)
	sizeData = len(data[0])
	evaluate = []
	for x in range(0,numVariables):
		auxValue = []
		for y in range(0,sizeData):
			auxValue.append(round(data[x][y]*linearRegressionModel[x+1],8))
		evaluate.append(auxValue)

	if flagLogistic:
		interval = []
		print(categoricalOutputLabels)
		auxInterval = 1/len(categoricalOutputLabels)
		for elem in categoricalOutputLabels:
			interval.append((elem,auxInterval))
			auxInterval += 1/len(categoricalOutputLabels)

	estimate = []
	for y in range(0,sizeData):
		auxEstimate = linearRegressionModel[0]
		for x in range(0,numVariables):
			auxEstimate += evaluate[x][y]

		if flagLogistic:
			auxEstimate = 1/(1 + math.exp(auxEstimate*-1))
			for elem in interval:
				if auxEstimate <= elem[1]:
					estimate.append(elem[0])
					break
		else:
			estimate.append(round(auxEstimate,4))

	return estimate

"""
- Función encargada de obtener el tipo de dato con el que se va a trabajar.
- Function in charge of obtaining the type of data with which to wor
"""
def getTypeData(data):
	flag = True
	auxType = type(data[0])
	for elem in data:
		if auxType != type(elem):
			flag = False
			break
	if flag:
		if auxType is str:
			return "Categorical"
		else:
			return "Numeric"
	else:
		return "Error"

"""
- Función encargada de obtener las etiquetas los datos categóricos.
- Function in charge of obtaining the labels of the categorical data.
"""
def getCategoricalLabels(data):
	categoricalLabels = [data[0]]
	for elem in data:
		if not elem in categoricalLabels:
			categoricalLabels.append(elem)
	return categoricalLabels

"""
- Función encargada de convertir los datos categóricos de salida en datos numéricos.
- Function in charge of converting the output categorical data into numerical data.
"""
def categoricalInput2Binary(data,order=[]):
	if order == []:
		order = [data[0]]
		for elem in data:
			if not elem in order:
				order.append(elem)

	dataOut = []
	for elem in order:
		auxData = []
		for elem2 in data:
			if elem == elem2:
				auxData.append(1.0)
			else:
				auxData.append(0.0)
		dataOut.append(auxData)
	return dataOut

"""
- Función encargada de convertir los datos categóricos de salida en datos numéricos.
- Function in charge of converting the output categorical data into numerical data.
"""
def categoricalOutput2Number(data):
	auxProb = round(np.log((data.count(data[0])/len(data))/(1-(data.count(data[0])/len(data)))),4)
	prob = [(data[0],auxProb)]
	for elem in data:
		auxCount = data.count(elem)
		flag = True
		for elem2 in prob:
			if elem2[0] == elem:
				flag = False
		
		if flag:
			auxProb = round(np.log((auxCount/len(data))/(1-(auxCount/len(data)))),4)
			prob.append((elem,auxProb))

	dataOut = []
	for elem in data:
		for elem2 in prob:
			if elem2[0] == elem:
				dataOut.append(elem2[1])

	return dataOut

"""
- Función encargada de realizar la etapa de preprocesamiento de los datos. (Transformando datos de entrada en numéricos)
- Function in charge of carrying out the data preprocessing stage. (Transforming input data into numbers)
"""
def preProcessingStage(data):
	transformedData = []
	categoricalInputLabels = []
	categoricalOutputLabels = []
	flagLogistic = False

	#Variables Independientes.
	for elemt in data[:-1]:
		auxType = getTypeData(elemt)
		if auxType == "Numeric":
			transformedData.append(elemt)
		elif auxType == "Categorical":
			transformedData += categoricalInput2Binary(elemt)
			categoricalInputLabels.append(getCategoricalLabels(elemt))
		else:
			print("\nError in struct of the data\n")
			exit()

	#Variable Dependiente.
	auxType = getTypeData(data[-1]) 
	if auxType == "Numeric":
		flagLogistic = False
		transformedData.append(data[-1])
	elif auxType == "Categorical":
		flagLogistic = True
		transformedData.append(categoricalOutput2Number(data[-1]))
		categoricalOutputLabels = getCategoricalLabels(data[-1])
	else:
		print("\nError in struct of the data\n")
		exit()

	return transformedData,categoricalInputLabels,categoricalOutputLabels,flagLogistic


"""
- Función encargada de guardar el modelo de regresión linear en un archivo json.
- Function in charge of saving the linear regression model in a json file.
"""
def saveModelLinearRegression(data):
	file = open("./LinearRegression.json", "w")
	model = "[\n\t{\n"
	for elem in data:
		model += "\t" + elem + ",\n"
	model = model[:-2]
	model += "\n\t}\n]"
	file.write(model)
	file.close()
"""
- Función encargada de realizar la etapa de procesamiento de los datos. (Construyendo el modelo de regresión linear)
- Function in charge of carrying out the data processing stage. (Building the linear regression model)
"""
def processingStage(transformedData,flagLogistic):
	#Número de Variables de la fuente de datos.
	numVariables = len(transformedData)
	#Tamaño de la fuente de datos.
	sizeData = len(transformedData[0])
	#Construyendo Polinomios Factorizados.
	factoredPolynomials = createPolynomials(transformedData,sizeData,numVariables)
	#Desarrollando los cuadrados de los Polinomios Factorizados.
	developedPolynomials = [squaredPolynomial(term) for term in factoredPolynomials]
	#Construllendo el Polinomio S
	polynomialS = buildPolynomialS(developedPolynomials)
	#Obteniendo el Polinomio S
	auxPolynomialS = getPolinomial(polynomialS,"Polynomial S")
	#Calculando las derivadas parciales del polinomio S.
	equations = [derivedPolynomial(polynomialS,x) for x in range(0,numVariables)]
	#Visualización del sistema de ecuaciones.
	auxEquations = getSystemEquations(equations,"Equations")
	#Calculando y resolviendo el sistema de ecuaciones.
	solvingEquations = solvingSystemEquations(equations,numVariables)
	#Calculando los coeficientes del modelo de regresión lineal(betas)
	betas = calculatingBetas(solvingEquations,numVariables)
	#Construllendo el modelo de regresión lineal.
	linearRegressionModel = betas[::-1]
	#Visualización del modelo de regresión lineal.
	auxModel = getLinearRegressionModel(linearRegressionModel,flagLogistic)
	#Guardando los datos del modelo de regresión lineal en un archivo.
	saveModelLinearRegression([auxModel,auxEquations,auxPolynomialS])
	#Regresando el modelo de regresión lineal.
	return linearRegressionModel

"""
- Función encargada de realizar la etapa de resultados. (Evaluar un nuevo set de datos en el modelo de regresión linear)
- Function in charge of carrying out the results stage. (Evaluate a new data set in the linear regression model)
"""
def resultsStage(linearRegressionModel,data,categoricalInputLabels,categoricalOutputLabels,flagLogistic,attributes):
	transformedData = []
	auxOrder = 0
	#Tranformando datos de entrada en númericos.
	for elemt in data:
		auxType = getTypeData(elemt)
		if auxType == "Numeric":
			transformedData.append(elemt)
		elif auxType == "Categorical":
			transformedData += categoricalInput2Binary(elemt,categoricalInputLabels[auxOrder])
			auxOrder += 1
		else:
			print("\nError in struct of the data\n")
			exit()

	#Evaluando el nuevo set de datos para estimar el valor de la variable dependiente.
	estimate = evaluateLinearRegressionModel(linearRegressionModel,transformedData,categoricalOutputLabels,flagLogistic)
	#Agregando la estimación a la tabla de datos de entrada.
	transformedData.append(estimate)
	#Visualización de los datos de entrada con su estimación.
	saveResults(transformedData,attributes,flagLogistic)


"""
- Función encargada de guardar los resultados en un archivo "csv".
- Function in charge of saving the results in a "csv" file.
"""
def saveResults(data,attributes,flagLogistic):
	file = open("./Predicciones.csv", "w")

	result = ""
	for x in range(0,len(attributes)-1):
		result +=  str(attributes[x]) + ","
	result += str(attributes[-1]) + "\n"

	sizeData = len(data[0])
	numVariables = len(data)
	
	for y in range(0,sizeData):
		for x in range(0,numVariables):
			if flagLogistic:
				if numVariables-1 == x:
					result += str(data[x][y])
				else:
					result += str(round(data[x][y],3)) + ","
			else:
				if numVariables-1 == x:
					result += str(round(data[x][y],3))
				else:
					result += str(round(data[x][y],3)) + ","
		result+= "\n"
		
	file.write(result+"\n")
	file.close()

"""
- Función encargada de cargar los datos almacenados en un archivo "csv" a memoria.
- Function in charge of loading the data stored in a "csv" file into memory.
"""
def readFileCsv(file):
	file = open(file)
	attributes = file.readline().strip().split(",")
	numColumns = len(attributes)
	data = []
	for x in range(0,numColumns):
		data.append([])

	for line in file:
		auxData = line.strip().split(",")
		for i in range(0,len(auxData)):
			auxElem = auxData[i].isdigit()
			if auxElem or "." in auxData[i] or "-" in auxData[i]:
				data[i].append(float(auxData[i]))
			else:
				attributes[i] = None
				data[i].append(auxData[i])

	file.close()
	return data,attributes


#-----------------------------------------Preparación de los Datos--------------------------------------------------#
parser = argparse.ArgumentParser(description='Regression Linear')
parser.add_argument('inDirTraining', type=str, help='Input directory for Training Data')
parser.add_argument('inDirTest', type=str, help='Input directory for test Data')
args = parser.parse_args()
	
#Leyendo archivo de la fuente de datos para la construcción del modelo de regresión linear.
data,attributes = readFileCsv(args.inDirTraining + ".csv")
#---------------------------------- Construcción del Modelo de Regresión Linear ------------------------------------#
#Tranformando datos de entrada en númericos.
transformedData,categoricalInputLabels,categoricalOutputLabels,flagLogistic = preProcessingStage(data)
#Construyendo el modelo de regresión linear.
linearRegressionModel = processingStage(transformedData,flagLogistic)
#---------------------------------- Pruebas sobre el Modelo de Regresión Linear ------------------------------------#
#Leyendo archivo de la fuente de datos para probar el modelo de regresión linear.
data,_ = readFileCsv(args.inDirTest + ".csv")
#Construllendo la lista de atributos.
newAttributes = []
auxCount = 0
for x in range(0,len(attributes)):
	if attributes[x] == None:
		if len(categoricalInputLabels) == 0:
			newAttributes.append("y")
		else:
			newAttributes = newAttributes + categoricalInputLabels[auxCount]
		auxCount += 1
	else:
		newAttributes.append(attributes[x])
#Probando el modelo de regresión linear.
resultsStage(linearRegressionModel,data,categoricalInputLabels,categoricalOutputLabels,flagLogistic,newAttributes)