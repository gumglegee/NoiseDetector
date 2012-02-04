import java.io.File;
import java.io.IOException;
import java.util.Random;

import rml.classifiers.Bagging;
import rml.io.IOprocess;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class NoiseDetector {

	public NoiseDetector() {
		super();
		// TODO Auto-generated constructor stub
	}

	public int[] calPrediction(Instances oriDataSet, String[] predictResults) {
		int[] results = new int[3];	//0:noise; 1:detected; 2:predict correctly
		for (int insIndex = 0; insIndex < oriDataSet.size(); insIndex++) {
			Instance ins = oriDataSet.get(insIndex);
			
			//check whether it is an artificial noise, true: is; false: not
			String oriLabel = oriDataSet.attribute(0).value((int) ins.value(0));
			String artLabel = oriDataSet.classAttribute().value((int) ins.classValue());
			boolean isArtNoise = !oriLabel.equals(artLabel);
			
			if(isArtNoise)
				results[0]++;
			
			//check whether the noise is detected by classifier, true: yes; false: no
			String predictLabel = predictResults[insIndex];
			boolean isDetected = !predictLabel.equals(artLabel);
			
			if(isDetected)
				results[1]++;
			
			//calculate perdiction
			if(! (isArtNoise ^ isDetected))
				results[2]++;
			
		}	// end for instance
		
		return results;
	}
	
	/**
	 * 
	 * @param trainingSet
	 * @throws Exception
	 */
/*	public double runBagging (Instances oriDataSet, Instances newDataSet) throws Exception {
		//bagging
		Bagging bagCla = new Bagging();
		
		//set the bag size 110% of original data set
		bagCla.setBagSizePercent(110);
		
		//set model as SMO
		SMO smo = new SMO();
		smo.setOptions(weka.core.Utils.splitOptions("-C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\""));
	//	smo.setOptions(weka.core.Utils.splitOptions("-C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.01\""));
		bagCla.setClassifier(smo);
		
		//train
		bagCla.buildClassifier(newDataSet);
		
		Evaluation eva = new Evaluation(newDataSet);
		int countNoise = 0;
		int countDetected = 0;
		for (Instance ins : newDataSet) {
			String predictClass = newDataSet.classAttribute().value((int)eva.evaluateModelOnce(bagCla, ins));
			int insIndex = newDataSet.indexOf(ins);
			Instance oriIns = oriDataSet.get(insIndex);
			String oriLabel = oriDataSet.attribute(0).value((int)oriIns.value(0));
			String noiseLabel = newDataSet.classAttribute().value((int)ins.value(newDataSet.numAttributes() - 1));
			
			//if artifically modified as noise
			//oriIns.value(0): original true class label
			//ins.value(newDataSet.numAttributes() - 1): noise-added class label
			if (!oriLabel.equals(noiseLabel)) {
			//	System.out.println(oriIns + " " + oriIns.value(0) + " " + ins.value(newDataSet.numAttributes() - 1));
				countNoise++;
				
				//if predicted class equals original class label, then this noise instance is detected
				if (predictClass.equals(oriLabel)) {
		//			System.out.println(oriIns.value(0) + " " + ins.value(newDataSet.numAttributes() - 1) + " "+ predictClass);
					countDetected++;
				}
			}
			
		//	if (newDataSet.classAttribute().value((int)ins.value(newDataSet.numAttributes() - 1)).equals(predictClass))
		//		count++;
			
		}
	//	System.out.println("Detected " + countDetected);
	//	System.out.println("Total Noise " + countNoise);
		
		System.out.print(countDetected + " " + countNoise + " ");
		
		return countDetected * 100.0 / countNoise;
	}
*/	
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

		try {
			String[] smoOptions = weka.core.Utils.splitOptions("-C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\"");
		//	String[] smoOptions = weka.core.Utils.splitOptions("-C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.01\"");
		
			//initial the file path
			String pathPrefix = args[0];
			int[] noiseLevels = {10, 20, 30, 40};
			String[] filePaths = new String[noiseLevels.length];
		
			for (int index = 0; index < noiseLevels.length; index++) {
				filePaths[index] = pathPrefix + "_" + noiseLevels[index] + ".csv";
				//	System.out.println(filePaths[index]);
			
				NoiseDetector rt = new NoiseDetector();


				//read the original data set as training set
				Instances oriDataSet = new IOprocess().readCSV(filePaths[index]);
			
				//TESTing only
		//		Instances oriDataSet = new IOprocess().readCSV("/h/hzhang07/assignment/COMP150-08/iris_ori.csv");
		//		System.out.println(oriDataSet.relationName());
				
				//TESTing only
		//		System.out.println(oriDataSet);
				
				//remove the first column refers to whether this instance is artifically modified as noise
				Instances newDataSet = new IOprocess().removeNoiseLabel(oriDataSet);
			
				//original weka bagging function
		//		System.out.print(noiseLevels[index] + " ");
		//		double precision = rt.runBagging(oriDataSet, newDataSet);
		//		System.out.println(precision);
				
				Bagging baggingModel = new Bagging();
				baggingModel.setNumIterations(10);
				baggingModel.setPercentReplicates(110);
				
				SMO smo = new SMO();
				smo.setOptions(smoOptions);
				
				baggingModel.buildClassifier(smo, newDataSet);
		//		int[][] voteResults = baggingModel.vote(newDataSet);
				String[] predictResults = baggingModel.classify(newDataSet);
				
		/*		for (int insIndex = 0; insIndex < newDataSet.size(); insIndex++) {
		//			System.out.println(newDataSet.get(insIndex) + " " + predictResults[insIndex]);
					System.out.print(newDataSet.get(insIndex) + " ");
					for (int voteResult : voteResults[insIndex])
						System.out.print(voteResult + " ");	//newDataSet.classAttribute().value(  )
					System.out.println();
				}
		*/		
				int[] precisionResults = rt.calPrediction(oriDataSet, predictResults);
				double precision = precisionResults[2] * 100.0 / oriDataSet.size();
				System.out.println(noiseLevels[index] + " " + precisionResults[0] + " " + precisionResults[1] + " " + precisionResults[2] + " " + precision);

			}	//end for noise level
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		System.out.println("read finish!");
	}

}
