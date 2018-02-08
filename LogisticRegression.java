import java.util.Random;

public class LogisticRegression {
	
	private double[] weights;
	private double[] initialWeights;
	private DataSet trainingData;
	private double stepSize;
	private int maxNumberOfIterations;
	private int numberOfIterations;
	
	
	public LogisticRegression(DataSet trainingData, int maxNumberOfIterations, double stepSize){
		this.trainingData = trainingData;
		this.maxNumberOfIterations = maxNumberOfIterations;
		this.stepSize = stepSize;
		this.weights = new double[trainingData.getAttributeNames().length];  //attributes include each datapoint + the classification; we need 1 weight for each datapoint + the bias weight
		this.initialWeights = new double[trainingData.getAttributeNames().length];
		randomizeWeights(weights);
		for(int i = 0; i < this.weights.length; i++) this.initialWeights[i] = this.weights[i];
	}
	
	public void train(){
		numberOfIterations = 1;
		
		for(;numberOfIterations < maxNumberOfIterations; numberOfIterations++){
//			System.out.println("Training Iteration " + numberOfIterations);
			updateWeights();
		}
		
		System.out.println("\nWeights");
		System.out.print("[");
		for(int i = 0; i < weights.length; i++){
			if(i == weights.length - 1) System.out.print(weights[i]);
			else System.out.print(weights[i] + ", ");
		}
		System.out.print("]\n\n");
		
	}
	
	private void updateWeights(){
		double[] gradientVector = new double[weights.length];
		
		//calculate each gradient component over all records
		for(int i = 0; i < gradientVector.length; i++){
			for(int j = 0; j < trainingData.getRecords().size(); j++){
				Record record = trainingData.getRecords().get(j);
				//calculate dot product of weights and inputs
				double weightedSum = 0;
				for(int k = 0; k < weights.length; k++){
					if(k == 0) weightedSum += weights[k];
					else weightedSum += weights[k] * record.getValues()[k-1];
				}
//				System.out.println("Weighted Sum: " + weightedSum);
				if(i == 0) gradientVector[i] += (record.getClassification() - (Math.pow(Math.E, weightedSum) / (1 + Math.pow(Math.E, weightedSum))));
				else gradientVector[i] += record.getValues()[i-1] * (record.getClassification() - (Math.pow(Math.E, weightedSum) / (1 + Math.pow(Math.E, weightedSum))));
			}
//			System.out.println("\n");
		}
		
		//print gradient
		System.out.println("Gradient");
		System.out.print("[");
		for(int i = 0; i < gradientVector.length; i++){
			if(i == gradientVector.length - 1) System.out.print(gradientVector[i]);
			else System.out.print(gradientVector[i] + ", ");
		}
		System.out.print("]\n");
		
		//update weights based on current gradient 
		for(int i = 0; i < weights.length; i++){
			weights[i] = weights[i] + stepSize * gradientVector[i];
		}
	}
	
	
	
	public double query(Double[] query){
		double probability = 0;
		double sum = 0;
		for(int i = 0; i < weights.length; i++){
			if(i == 0) sum += weights[i];
			else sum += weights[i] * query[i-1];
		}
		sum *= -1;
		probability = Math.pow(Math.E, sum);
		probability += 1;
		return (1/probability);
	}
	
	private static void randomizeWeights(double[] weights){
		Random random = new Random();
		for(int i = 0; i < weights.length; i++) weights[i] = random.nextGaussian();
	}

}
