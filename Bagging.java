package rml.classifiers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import weka.classifiers.functions.SMO;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class Bagging {

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
	 * 
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
		int bagSize = dataSet.size() * percentReplicates / 100;
		
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
