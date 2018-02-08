
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;

public class Main{
	
	private static int maxNumberOfEpochs = 2000;
	private static double stepSize = .01;
	
	private static String dataPath = "/Users/jmccune/Desktop/training_logistic.csv";
	
	public static void main(String[] args) {
		FileReader trainDataFile;
		LogisticRegression logisticRegression;
		
		Scanner scanner = new Scanner(System.in);
		
		do {
			System.out.println("Input path to training data file.");
			try {
//				trainDataFile = new FileReader(new File(scanner.nextLine()));
				trainDataFile = new FileReader(new File(dataPath));
				break;
			} catch (FileNotFoundException fnfe) {
				System.out.println("File Not Found.");
				continue;
			}
		} while (true);
		scanner.close();
		
		DataSet trainingData = new DataSet("TRAINING", trainDataFile);
		try {
			trainingData.parseCSV();
			System.err.println(trainingData.toString());
		} catch (IOException e) {
			System.err.println("Problem Parsing Training Data.");
		}
		logisticRegression = new LogisticRegression(trainingData, maxNumberOfEpochs, stepSize);
		logisticRegression.train();
		System.out.println("Probability of failure for 3 is: " + logisticRegression.query(new Double[]{3.}));
		System.out.println("Probability of failure for 5 is: " + logisticRegression.query(new Double[]{5.}));
	}
}
