package rml.classifiers;

import java.awt.Point;
import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.Callable;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instance;
import weka.core.Instances;

public class OptionVal extends Thread {

	private Instances randData;
	private int foldNum;
	private int foldIndex;
	
	private ArrayList<Point> grid;
	private Point p;
	
	private double[] accRates;
	private double accRate;
	
	public void setPrep(int foldNum, int foldIndex, Instances randData, ArrayList<Point> grid) {
		this.foldNum = foldNum;
		this.foldIndex = foldIndex;
		this.randData = randData;
		this.grid = grid;
		
		accRates = new double[grid.size()];
	}
	
	public void setPrep(int foldNum, Instances randData, Point p) {
		this.foldNum = foldNum;
		this.randData = randData;
		this.p = p;
	}
/*	
	public void run() {
		//TESTing olny
		System.out.println("Start " + foldIndex);
		
		//split training set and test set
		Instances train = randData.trainCV(foldNum, foldIndex);
		Instances test = randData.testCV(foldNum, foldIndex);
		
		//for each point in the grid, set the parameters of SMO and calculate the accuracy or classification
//		GridPointVal[] gpvThreads = new GridPointVal[grid.size()];
		for (int pointIndex = 0; pointIndex < grid.size(); pointIndex++) {
			//set the paramaters of SMO and kernel
			Point p = grid.get(pointIndex);
			try {
				RBFKernel rbf = new RBFKernel();
				rbf.setOptions(weka.core.Utils.splitOptions("-G " + Math.pow(2, p.y)));
				
				SMO smo = new SMO();
				smo.setC(Math.pow(2, p.x));
				smo.setKernel(rbf);
				smo.buildClassifier(train);
	
				accRates[pointIndex] = calAcc(smo, test);

			} catch (Exception e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
		}
		
		//TESTing olny
		System.out.println("Finish " + foldIndex);
	}
*/
	
	public void run() {
		try {
			RBFKernel rbf = new RBFKernel();
			
			rbf.setOptions(weka.core.Utils.splitOptions("-G " + Math.pow(2, p.y)));
			SMO smo = new SMO();
			smo.setC(Math.pow(2, p.x));
			smo.setKernel(rbf);

			Evaluation eva = new Evaluation(randData);
			eva.crossValidateModel(smo, randData, foldNum, new Random());
			
			accRate = eva.pctCorrect();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
	}
	
	private double calAcc(SMO smo, Instances testDataSet) throws Exception {
		int mislabeled = 0;
		for (Instance ins : testDataSet) {
			int predictClassIndex = (int) smo.classifyInstance(ins);
			int oriClassIndex = (int) ins.classValue();
			
			if (predictClassIndex != oriClassIndex)
				mislabeled++;
		}
			
		double accuracy = 100.0 - mislabeled * 100.0 / testDataSet.size();
		
		return accuracy;
	}
	
	public double[] getAccRates() {
		return accRates;
	}

	public double getAccRate() {
		return accRate;
	}
}
