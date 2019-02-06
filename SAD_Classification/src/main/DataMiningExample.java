/*
 GOAL: Load data from .arff files, preprocess the data, train a model and assess it either using 10-fold cross-validation or hold-out
 
 Compile:
 javac DataMiningExample.java

 Run Interpret:
 java DataMiningExample
 
 HACER!!!
	- Hacer modular
	- El programa no puede tener dependencias con datos!
	- Generar un .jar y ejecutar desde la línea de comandos

 */
package main;

import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.instance.Randomize;
import java.util.Random;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class DataMiningExample {

	public static void main(String[] args) throws Exception {
		/////////////////////////////////////////////////////////////
		// 1. LOAD DATA FILE
		// HACER!!!! Bloque 1: como sub-clase
		// 1.1. Get the path of the .arff (instances) from the command line
		/*
		 * if( args.length < 1 ){ System.out.
		 * println("OBJETIVO: Seleccionar atributos (AttributeSelection<-CfsSubsetEval, search<-BestFirst) y Evaluar clasificador NaiveBayes con 10-fold cross-validation."
		 * ); System.out.println("ARGUMENTOS:");
		 * System.out.println("\t1. Path del fichero de entrada: datos en formato .arff"
		 * ); return; }
		 */
		// 1.2. Open the file

		DataSource source = new DataSource("resources/heart-c.arff");
		Instances data = source.getDataSet();

		// 1.5. Shuffle the instances: apply Randomize filter
		Randomize filtro = new Randomize();
		filtro.setInputFormat(data);
		filtro.setRandomSeed(0);
		data = Filter.useFilter(data, filtro);

		// 1.6. Specify which attribute will be used as the class: the last one, in this
		// case
		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);

		/////////////////////////////////////////////////////////////
		// 2. FEATURE SUBSET SELECTION
		// HACER!!!! Empaquetar Bloque 2: como sub-clase
		AttributeSelection filter = new AttributeSelection();
		CfsSubsetEval eval = new CfsSubsetEval();
		BestFirst search = new BestFirst();
		filter.setEvaluator(eval);
		filter.setSearch(search);
		filter.setInputFormat(data);
		// 2.1 Get new data set with the attribute sub-set
		Instances newData = Filter.useFilter(data, filter);
		
		

		/////////////////////////////////////////////////////////////
		// 3. CLASSIFY:

		// 3.0 Train the classifier (estimador) by means of: the Naive Bayes algorithm
		// (in this case)
		//NaiveBayes estimador = new NaiveBayes();// Naive Bayes
		
		IBk classifier = new IBk(3);

		// 3.1 Assess the performance of the classifier by means of 10-fold
		// cross-validation
		// HACER!!!! Empaquetar Bloque 3.1: como sub-clase
		Evaluation evaluatorHO = evalHoldOut(classifier, newData, 70.0);
		getResults(evaluatorHO);
		
		Evaluation evaluatorNH = evalNoHonesta(classifier, newData);
		getResults(evaluatorNH);
		
		Evaluation evaluatorKFCV = evalKFCV(classifier, newData, 10);
		getResults(evaluatorKFCV);
		
		Evaluation evalLeave1Out = evalLeaveOneOut(classifier, newData);
		getResults(evalLeave1Out);
		
		//Evaluation evaluator = new Evaluation(train);
		//evaluator.crossValidateModel(classifier, test, 10, new Random(1)); // Random(1): the seed=1 means "no shuffle"
																				// :-!
		// HACER!!!! Imprimir matriz de confusión
		

		/*
		 * 
		 * 
		 * // 3.2.c Let the model predict the class for each instance in the test set
		 * evaluator.evaluateModel(estimador, test); double predictions[] = new
		 * double[test.numInstances()]; for (int i = 0; i < test.numInstances(); i++) {
		 * predictions[i] =
		 * evaluator.evaluateModelOnceAndRecordPrediction((Classifier)estimador,
		 * test.instance(i)); } // HACER!!!! Guardar en un fichero de salida la clase
		 * estimada por el modelo para cada instancia del test y así después podremos
		 * comparar la clase real y la estimada
		 * 
		 */
		///////////////////////////////////////////////////////
		// Observa: http://weka.wikispaces.com/Use+Weka+in+your+Java+code
		///////////////////////////////////////////////////////

	}
	
	private static Evaluation evalNoHonesta(Classifier pClassifier, Instances pData) throws Exception {
        // se instancia el evaluador
        Evaluation evaluator = new Evaluation(pData);
        
        // se entrena el clasificador con el set entero de datos
        pClassifier.buildClassifier(pData);
        
        // se evalua el clasificador con el set entero de datos
        evaluator.evaluateModel(pClassifier, pData);
        
        // se devuelven los resultados
        return evaluator;
    }
    
    private static Evaluation evalHoldOut(Classifier pClassfier, Instances pData, double pTrainPercent) throws Exception {
        // se instancia el evaluador
        Evaluation evaluator = new Evaluation(pData);
        
        // se calcula el número de instancias de entrenamiento y de test en base al porcentaje
        int numInstances = pData.numInstances();
        int numTrain = (int) (numInstances * pTrainPercent / 100);
        int numTest = numInstances - numTrain;
        
        // se obtienen los conjuntos de entrenamiento y de test
        pData.randomize(new Random(1));
        Instances trainData = new Instances(pData, 0, numTrain);
        Instances testData = new Instances(pData, numTrain, numTest);
        
        // se entrena el clasificador
        pClassfier.buildClassifier(trainData);
        
        // se evalua el clasificador
        evaluator.evaluateModel(pClassfier, testData);
        
        //se devuelven los resultados
        return evaluator;
    }
    
    private static Evaluation evalKFCV(Classifier pClassifier, Instances pData, int pK) throws Exception {
        // se inicializa el evaluador con los datos para que reconozca el formato
        Evaluation evaluator = new Evaluation(pData);
        // se evalua el clasificador con el metodo k-fold cross validation
        evaluator.crossValidateModel(pClassifier, pData, pK, new Random(1));
        // se escriben los resultados
        return evaluator;
    }

    private static Evaluation evalLeaveOneOut(Classifier pClassifier, Instances pData) throws Exception {
        // se obtiene el número de instancias de los datos
        int k = pData.numInstances();
        // se hace el k-fold cross validation con k = num. instancias
        return evalKFCV(pClassifier, pData, k);
}
    
    
    private static void getResults(Evaluation evaluator) throws Exception {
    	
    	double acc = evaluator.pctCorrect();
		double inc = evaluator.pctIncorrect();
		double kappa = evaluator.kappa();
		double mae = evaluator.meanAbsoluteError();
		double rmse = evaluator.rootMeanSquaredError();
		double rae = evaluator.relativeAbsoluteError();
		double rrse = evaluator.rootRelativeSquaredError();
		double confMatrix[][] = evaluator.confusionMatrix();

		System.out.println("Correctly Classified Instances  " + acc);
		System.out.println("Incorrectly Classified Instances  " + inc);
		System.out.println("Kappa statistic  " + kappa);
		System.out.println("Mean absolute error  " + mae);
		System.out.println("Root mean squared error  " + rmse);
		System.out.println("Relative absolute error  " + rae);
		System.out.println("Root relative squared error  " + rrse);
		System.out.println("");
		
		StringBuilder resultado = new StringBuilder();
      
		
		resultado.append(evaluator.toSummaryString());
    	resultado.append("\n");
    	resultado.append(evaluator.toClassDetailsString());
    	resultado.append("\n");
    	resultado.append(evaluator.toMatrixString());
    	resultado.append("\n");
    	
    	System.out.println(resultado);
    }
}
