package rml.classifiers;

import java.awt.Point;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class Bagging {

	private static int FOLDNUM = 10;
	
	private int percentReplicates = 100;

	private int numIterations = 10;
	
	private int voteThreshold;
	
	ArrayList<SMO> modelLists;
	
	public Bagging() {
		// TODO Auto-generated constructor stub
		modelLists = new ArrayList<SMO>();
	}
	
	public Bagging(int percentReplicates, int numIterations) {
		this.percentReplicates = percentReplicates;
		this.numIterations = numIterations;
		
	}

	/**
	 * Build the classifier with optimal parameters by grid search
	 * @param dataSet
	 * @throws Exception
	 */
	public void buildClassifier(Instances dataSet) throws Exception {
		for (int indexIter = 0; indexIter < numIterations; indexIter++) {
			Instances trainingSet = generateBagTrainingSet(dataSet);
			
			//TESTing only
	//		System.out.println("Start parameter selection for Bag " + indexIter);
	//		Point bestPara = gridSearch(smo, dataSet);
	//		System.out.println("Optimal parameter for Bag " + indexIter + ": "+ bestPara.x + " " + bestPara.y);
			
			//grid search to get the optimal parameters for classifier
			Point bestPara = gridSearchCV(dataSet);
		
			SMO smo = getBestClassifier(bestPara);
			smo.buildClassifier(trainingSet);
			
			//add this model in the model list
			modelLists.add(smo);	
		}	//end for bagging iteration
		
	}
	
	/**
	 * Build the classifier with arbitrary settings
	 * @param smo
	 * @param dataSet
	 * @throws Exception
	 */
	public void buildClassifier(SMO smo, Instances dataSet) throws Exception {
		for (int indexIter = 0; indexIter < numIterations; indexIter++) {
			Instances trainingSet = generateBagTrainingSet(dataSet);
			
			smo.buildClassifier(trainingSet);
			
			//add this model in the model list
			modelLists.add(smo);	
		}	//end for bagging iteration
		
	}
	
	/**
	 * classify the test data and return the votes of bagging for each instance
	 * @param testDataSet
	 * @return
	 * @throws Exception
	 */
	public int[][] vote(Instances testDataSet) throws Exception {
		int[][] voteResults = new int[testDataSet.size()][testDataSet.classAttribute().numValues()];
		
		for (int indexIter = 0; indexIter < numIterations; indexIter++) {
			//get the model for testing in this iteration
			SMO iterModel = modelLists.get(indexIter);
			
			//classify instances in test set one by one
			for (Instance ins : testDataSet) {
			//	System.out.print(ins + " ");
				double[] fDistribution = iterModel.distributionForInstance(ins);
				int predictClassIndex = (int) iterModel.classifyInstance(ins);
				int insIndex = testDataSet.indexOf(ins);
				voteResults[insIndex][predictClassIndex]++;
				
				//test only: output voting result
			/*	for (double d : fDistribution) {
					int classVote = (int) (d * totalVote);
					System.out.print(d + " ");
				}
				System.out.println();
			*/
			
			}	//end for test instance
			
		}	//end for bagging iteration
		
		return voteResults;
	}
	
	/**
	 * classify the test data and return the predicted class label for each instance
	 * @param testDataSet
	 * @return
	 * @throws Exception
	 */
	public String[] classify(Instances testDataSet) throws Exception {
		int[][] voteResults = vote(testDataSet);
		String[] predictResults = new String[testDataSet.size()];
		
		for (int insIndex = 0; insIndex < voteResults.length; insIndex++) {
			predictResults[insIndex] = new String(testDataSet.classAttribute().value(findMax(voteResults[insIndex])));
		}
		
		return predictResults;
	}
	
	private Instances generateBagTrainingSet(Instances dataSet) {
		int bagSize = (int) (dataSet.size() * percentReplicates * 1.0 / 100.0);
		
		//init trainingSet
		String relationName = dataSet.relationName() + "_bag";
		ArrayList<Attribute> bagAttr = new ArrayList<Attribute>();
		for (int index = 0; index < dataSet.numAttributes(); index++)
			bagAttr.add(index, dataSet.attribute(index));
		Instances trainingSet = new Instances(relationName, bagAttr, 0);
		
		//random select instance from original data set
		Random generator = new Random();
		for (int index = 0; index < bagSize; index++) {
			int insIndex = generator.nextInt(dataSet.size());
			trainingSet.add(dataSet.get(insIndex));
		}	//end for bag size
		
		//IMPORTANT: set the index of class attribute
		//if the index is not preset, exception will be reported
		if (trainingSet.classIndex() == -1)
			trainingSet.setClassIndex(trainingSet.numAttributes() - 1);
		
		return trainingSet;
	}
	
	private int findMax(int[] oriArray) {
		int largest = oriArray[0];
		int indexMax = 0;
		for (int index = 1; index < oriArray.length; index++) {
			if (oriArray[index] >= largest) {
				largest = oriArray[index];
				indexMax = index;
			}
		}
		
		return indexMax;
	}
	
	private int findMax(double[] oriArray) {
		double largest = oriArray[0];
		int indexMax = 0;
		for (int index = 1; index < oriArray.length; index++) {
			if (oriArray[index] >= largest) {
				largest = oriArray[index];
				indexMax = index;
			}
		}
		
		return indexMax;
	}
	
	private Point gridSearch(SMO smo, Instances dataSet) throws Exception {
		//generate grid
		ArrayList<Point> grid = new ArrayList<Point>();
		for (int indexC = 15; indexC > -6; indexC -= 2) {
			for (int indexGamma = 3; indexGamma > -16; indexGamma -= 2) {
				Point point = new Point();
				point.x = indexC;
				point.y = indexGamma;
				
				grid.add(point);
			}
		}
		
		
		Random rand = new Random();
		Instances randData = new Instances(dataSet);
		randData.randomize(rand);
		
		OptionVal[] optThreads = new OptionVal[Bagging.FOLDNUM];
		for (int foldIndex = 0; foldIndex < Bagging.FOLDNUM; foldIndex++) {
			optThreads[foldIndex] = new OptionVal();
			optThreads[foldIndex].setPrep(Bagging.FOLDNUM, foldIndex, randData, grid);
			optThreads[foldIndex].start();
		}	//for foldIndex
		
		for (int foldIndex = 0; foldIndex < Bagging.FOLDNUM; foldIndex++)
			optThreads[foldIndex].join();
		
		double[] accRates = new double[grid.size()];
		for (int pointIndex = 0; pointIndex < grid.size(); pointIndex++)
			for (int foldIndex = 0; foldIndex < Bagging.FOLDNUM; foldIndex++) {
				accRates[pointIndex] += optThreads[foldIndex].getAccRates()[pointIndex];
			}
		
		//TESTing only
	//	for (int pointIndex = 0; pointIndex < grid.size(); pointIndex++)
	//		System.out.println(grid.get(pointIndex).x + " " + grid.get(pointIndex).y + " " + accRates[pointIndex]);
		
		int maxIndex = findMax(accRates);
		
	//	System.gc();
		
		return grid.get(maxIndex);
	}
	
	public Point gridSearchCV(Instances dataSet) throws InterruptedException {
		//generate grid
		ArrayList<Point> grid = new ArrayList<Point>();
		for (int indexC = 15; indexC > -6; indexC -= 2) {
			for (int indexGamma = 3; indexGamma > -16; indexGamma -= 2) {
				Point point = new Point();
				point.x = indexC;
				point.y = indexGamma;
				
				grid.add(point);
			}
		}
		
		OptionVal[] optThreads = new OptionVal[grid.size()];
		for (int pointIndex = 0; pointIndex < grid.size(); pointIndex++) {
			Point p = grid.get(pointIndex);
			
			optThreads[pointIndex] = new OptionVal();
			optThreads[pointIndex].setPrep(Bagging.FOLDNUM, dataSet, p);
			optThreads[pointIndex].start();
		}
		
		for (int pointIndex = 0; pointIndex < grid.size(); pointIndex++)
			optThreads[pointIndex].join();
		
		double[] accRates = new double[grid.size()];
		for (int pointIndex = 0; pointIndex < grid.size(); pointIndex++)
			accRates[pointIndex] = optThreads[pointIndex].getAccRate();
		
		int maxIndex = findMax(accRates);
		
		System.gc();
			
		return grid.get(maxIndex);
	}
	
	private SMO getBestClassifier(Point bestPara) throws Exception {
		RBFKernel rbf = new RBFKernel();
		rbf.setOptions(weka.core.Utils.splitOptions("-G " + Math.pow(2, bestPara.y)));
		SMO smo = new SMO();
		smo.setC(Math.pow(2, bestPara.x));
		smo.setKernel(rbf);
		
		return smo;
	}
	
	public int getPercentReplicates() {
		return percentReplicates;
	}
	
	public void setPercentReplicates(int percentReplicates) {
		this.percentReplicates = percentReplicates;
	}

	public int getNumIterations() {
		return numIterations;
	}

	public void setNumIterations(int numIterations) {
		this.numIterations = numIterations;
	}

}
