"""
Español
Autor: German Lopez Rodrigo.
Clase: Machine Learning.
Grupo: 01.
Versión: 1.0
Descripción:
El siguiente programa crea un modelo de Naiave Bayes, recibe dos archivos tipo csv
uno para crear el modelo y otro para hacer las pruebas del modelo. En el segúndo archivo
se debe quitar la variable dependiente.

English
Author: German Lopez Rodrigo.
Class: Machine Learning.
Group: 01.
Version: 1.0
Description:
The following program creates a model of Naiave Bayes, receives two csv files
one to create the model and another to test the model. In the second file
the dependent variable must be removed.
"""

import argparse
import math

"""
- Función encargada de obtener el tipo de dato que maneja cada clase de la fuente de datos.
- Function in charge of obtaining the type of data that each class of the data source handles.
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

		print("\nError in struct of the data\n")
		exit()

"""
- Función encargada de obtener los valores sin repetición de cada clase de la fuente de datos.
- Function in charge of obtaining the values ​​without repetition of each class from the data source.
"""
def getClassesValues(data,attributes):
	classesValues =[]
	for classValues in data:
		values = [classValues[0]]
		for elem in classValues:
			if elem not in values:
				values.append(elem)
		classesValues.append(values)
	return classesValues

"""
- Función encargada de calcular las probabilidades a priori del atributo a clasificar.
- Function in charge of calculating the a priori probabilities of the attribute to classify.
"""
def calculateProbabilitys(data,nameClass,indexClass,classesValues):
	probabilitys = {}
	sizeData = len(data[indexClass])
	for elem in classesValues:
		auxProb = round(data[indexClass].count(elem)/sizeData,4)
		label = "P(" + str(nameClass) + "=" + str(elem) + ")"
		probabilitys.setdefault(label,auxProb)
	return probabilitys

"""
- Función encargada de calcular la media de una Variable Aleatoria.
- Function in charge of calculating the mean of a Random Variable.
"""
def getMean(data,sizeData):
	mean = 0
	for x in range(0,len(data)):
		mean += data[x]
	return round(mean/sizeData,4)

"""
- Función encargada de calcular la desviación estándar de una Variable Aleatoria.
- Function in charge of calculating the standard deviation of a Random Variable.
"""
def getDeviation(data,mean,sizeData):
	deviation = 0
	for x in range(0,len(data)):
		deviation += (data[x]-mean)**2
	deviation = math.sqrt((1/(sizeData))*deviation)
	return round(deviation,4)

"""
- Función encargada de calcular la desviación estándar de una Variable Aleatoria.
- Function in charge of calculating the standard deviation of a Random Variable.
"""
def conditionalProbability(data,classes,attributes,typeAttributes,indexClass):
	cProbabilitys ={}
	for index in range(0,len(classes)-1):
		for attribute in classes[index]:
				if typeAttributes[index] == "Categorical":
					for attributeClass in classes[indexClass]:
						sizeData = data[indexClass].count(attributeClass)
						auxCount = len([x for x in range(0,len(data[index])) if (data[index][x] == attribute and data[indexClass][x] == attributeClass)])/sizeData
						label = "P(" + str(attributes[index]) + "=" + str(attribute) + "|" + str(attributes[indexClass]) + "="  + str(attributeClass) + ")"
						cProbabilitys.setdefault(label,round(auxCount,4))
				elif typeAttributes[index] == "Numeric":
					for attributeClass in classes[indexClass]:
						sizeData = data[indexClass].count(attributeClass)
						auxData = [data[index][x] for x in range(0,len(data[index])) if (data[indexClass][x] == attributeClass)]
						mean = getMean(auxData,sizeData)
						deviation = getDeviation(auxData,mean,sizeData)
						label = "N(" + str(attributes[index])  + "|" + str(attributes[indexClass]) + "="  + str(attributeClass) + ")"
						cProbabilitys.setdefault(label,(mean,deviation))
					break
	return cProbabilitys

"""
- Función encargada de guardar la clasificación de los datos en un archivo csv.
- Function in charge of saving the classification of the data in a csv file.
"""
def saveFileCsv(data,attributes,file,results):
	file = open(file+"Results.csv", "w")
	data.append(results)
	result = ""

	for elem in attributes:
		result += elem + ","
	result = result[:-1] + "\n"

	for indexX in range(0,len(results)):
		for indexY in range(0,len(data)):
			result += str(data[indexY][indexX]) + ","
		result = result[:-1] + "\n"
	file.write(result+"\n")
	file.close()


"""
- Función encargada de guardar las probabilidades calculadas en el entrenamiento del modelo en un archivo json.
- Function in charge of saving the classification of the data in a csv file.
"""
def saveFileJson(aprioriProbabilities,posterioriProbabilities,file):
	file = open(file+"Probabilitys.json", "w")
	probabilitys = "{\n"
	for key in aprioriProbabilities.keys():
		probabilitys += "\t\"" + str(key)  + "\": " + str(aprioriProbabilities.get(key)) + ",\n"

	for key in posterioriProbabilities.keys():
		aux = str(posterioriProbabilities.get(key)).replace('(','[').replace(')',']')
		probabilitys += "\t\"" + str(key)  + "\": " + aux + ",\n"

	probabilitys = probabilitys[:-2] + "\n"
	file.write(probabilitys+"}\n")
	file.close()

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
	for x in range(0,numColumns):
		data.append([])

	for line in file:
		auxData = line.strip().split(",")
		for i in range(0,len(auxData)):
			auxElem = auxData[i].isdigit()
			if auxElem or "." in auxData[i] or "-" in auxData[i]:
				data[i].append(float(auxData[i]))
			else:
				data[i].append(auxData[i])

	file.close()

	for x in range(0,numColumns): 
		typeAttributes.append(getTypeData(data[x]))

	return data,attributes,typeAttributes


#-----------------------------------------Preparación de los Datos--------------------------------------------------#
parser = argparse.ArgumentParser(description='Naive Bayes')
parser.add_argument('inDirTraining', type=str, help='Input directory for Training Data')
parser.add_argument('inDirTest', type=str, help='Input directory for test Data')
args = parser.parse_args()

#------------------------------------ Construcción del Modelo de Naive Bayes ---------------------------------------#
#Leyendo archivo de la fuente de datos para la construcción del modelo de Naive Bayes.
data,attributesTraining,typeAttributes = readFileCsv(args.inDirTraining + ".csv")
#Indice de la clase a clasificar.
classIndex  = len(attributesTraining)-1
#Obteniendo los valores sin repetición de cada clase de la fuente de datos.
classesValues = getClassesValues(data,attributesTraining)
#Calculando las probabilidades a priori de la clase a clasificar.
aprioriProbabilities = calculateProbabilitys(data,attributesTraining[classIndex],classIndex,classesValues[classIndex])
#Calculando las probabilidades a posteriori de las clases. (condicionales)
posterioriProbabilities = conditionalProbability(data,classesValues,attributesTraining,typeAttributes,classIndex)
#Probabilidades a priori de la clase a clasificar.
saveFileJson(aprioriProbabilities,posterioriProbabilities,args.inDirTraining)

#------------------------------------ Evaluación del Modelo de Naive Bayes ----------------------------------------#
#Leyendo archivo de la fuente de datos para probar el modelo de Naive Bayes.
data,attributesTest,typeAttributes = readFileCsv(args.inDirTest + ".csv")
#Evaluando el modelo de Naive Bayes para obtener la clasificación correpondiente a los datos de entrada.
results = [] 
for indexX in range(0,len(data[0])):
	result = []
	for attributeClass in classesValues[classIndex]:
		label = "P(" + str(attributesTraining[classIndex]) + "=" + str(attributeClass) + ")"
		prob = aprioriProbabilities.get(label)
		for indexY in range(0,len(attributesTest)):
			if typeAttributes[indexY] == "Categorical":
				label = "P(" + str(attributesTest[indexY]) + "=" + str(data[indexY][indexX]) + "|" + str(attributesTraining[classIndex]) + "="  + str(attributeClass) + ")"
				prob *= posterioriProbabilities.get(label)
					
			elif typeAttributes[indexY] == "Numeric":
				label = "N(" + str(attributesTest[indexY])  + "|" + str(attributesTraining[classIndex]) + "="  + str(attributeClass) + ")"
				med,deviation = posterioriProbabilities.get(label)
				auxProb = (1/(math.sqrt(2*math.pi)*deviation))*pow(math.e,(-((data[indexY][indexX]-med)**2)/(2*(deviation**2))))
				prob *= auxProb	
				
		result.append(prob)
	results.append(classesValues[classIndex][result.index(max(result))])
#Obteniendo las etiquetas de las clases.
attributesTest.append(attributesTraining[classIndex])
#Almacenando los resultados de clasificación de los datos de prueba.
saveFileCsv(data,attributesTest,args.inDirTraining,results)

