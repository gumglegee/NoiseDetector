package rml.io;

import java.io.File;
import java.io.IOException;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class IOprocess {

	public IOprocess() {
		// TODO Auto-generated constructor stub
	}
	
	/**
	 * 
	 * @param filePath
	 * @return
	 * @throws Exception
	 */
	public Instances readCSV(String filePath) throws Exception {
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File(filePath));
		
		//CSVLoader assumes that the first row in the file determines the number of and names of the attributes.
		//add '-H' option to CSVLoader, which refers to 'No header row present in the data'
		String[] oriLoaderOpt = loader.getOptions();
		String[] newLoaderOpt = new String[oriLoaderOpt.length + 1];
		for (int index = 0; index < oriLoaderOpt.length; index++)
			newLoaderOpt[index] = oriLoaderOpt[index];
		newLoaderOpt[newLoaderOpt.length-1] = "-H";
		loader.setOptions(newLoaderOpt);
		
		Instances data = loader.getDataSet();
		
		// setting class attribute if the data format does not provide this information
		// E.g., the XRFF format saves the class attribute information as well
		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		
		return data;
	}
	
	/**
	 * 
	 * @param data
	 * @param outPath
	 * @throws IOException
	 */
	public void outARFF(Instances data, String outPath) throws IOException {
	    ArffSaver saver = new ArffSaver();
	    saver.setInstances(data);
	    saver.setFile(new File(outPath));
	    saver.setDestination(new File(outPath));
	    saver.writeBatch();
	}

	/**
	 * remove the first column refers to whether this instance is artifically modified as noise
	 * @param oriDataSet
	 * @return
	 * @throws Exception
	 */
	public Instances removeNoiseLabel(Instances oriDataSet) throws Exception {
		Remove remove = new Remove();
		remove.setOptions(weka.core.Utils.splitOptions("-R 1"));
		remove.setInputFormat(oriDataSet);
		Instances newDataSet = Filter.useFilter(oriDataSet, remove);
		
		return newDataSet;
	}

}
