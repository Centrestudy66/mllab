import java.io.*;
import java.util.*;

public class NaiveBayesClassifier {
    private Map<String, Integer> categoryCounts = new HashMap<>();
    private Map<String, Map<String, Integer>> featureCounts = new HashMap<>();
    private int totalDocs = 0;
    private List<String> categories = Arrays.asList("Sports", "Politics", "Tech", "Business", "Entertainment");

    public void train(List<String[]> data) {
        for (String[] values : data) {
            String category = values[15]; // Category column

            categoryCounts.put(category, categoryCounts.getOrDefault(category, 0) + 1);
            totalDocs++;

            for (int i = 2; i <= 14; i++) { // Features from Word_Count to Contains_Email
                String key = "F" + i + "_" + values[i];
                featureCounts.putIfAbsent(category, new HashMap<>());
                featureCounts.get(category).put(key, featureCounts.get(category).getOrDefault(key, 0) + 1);
            }
        }
    }

    public String classify(String[] features) {
        double maxProb = Double.NEGATIVE_INFINITY;
        String bestCategory = null;

        for (String category : categories) {
            double prob = Math.log(categoryCounts.getOrDefault(category, 1) / (double) totalDocs);

            for (int i = 0; i < features.length; i++) {
                String key = "F" + (i + 2) + "_" + features[i];
                int count = featureCounts.getOrDefault(category, new HashMap<>()).getOrDefault(key, 1);
                prob += Math.log(count / (double) (categoryCounts.get(category) + 1));
            }

            if (prob > maxProb) {
                maxProb = prob;
                bestCategory = category;
            }
        }
        return bestCategory;
    }

    public static void main(String[] args) throws IOException {
        NaiveBayesClassifier classifier = new NaiveBayesClassifier();

        // Read dataset
        List<String[]> data = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new FileReader("dataset.csv"));
        String line = reader.readLine(); // Skip header
        while ((line = reader.readLine()) != null) {
            data.add(line.split(","));
        }
        reader.close();

        // Split dataset: 80% training, 20% testing
        Collections.shuffle(data, new Random());
        int trainSize = (int) (0.8 * data.size());

        List<String[]> trainData = data.subList(0, trainSize);
        List<String[]> testData = data.subList(trainSize, data.size());

        classifier.train(trainData);

        // Evaluate model
        int correct = 0;
        int total = testData.size();
        Map<String, Integer> truePositive = new HashMap<>();
        Map<String, Integer> falsePositive = new HashMap<>();
        Map<String, Integer> falseNegative = new HashMap<>();

        for (String[] testInstance : testData) {
            String actualCategory = testInstance[15];
            String predictedCategory = classifier.classify(Arrays.copyOfRange(testInstance, 2, 14));

            if (actualCategory.equals(predictedCategory)) {
                correct++;
                truePositive.put(actualCategory, truePositive.getOrDefault(actualCategory, 0) + 1);
            } else {
                falsePositive.put(predictedCategory, falsePositive.getOrDefault(predictedCategory, 0) + 1);
                falseNegative.put(actualCategory, falseNegative.getOrDefault(actualCategory, 0) + 1);
            }
        }

        // Calculate accuracy, precision, recall, and F1-score
        double accuracy = (double) correct / total;
        System.out.println("\n====== Model Evaluation ======");
        System.out.println("Accuracy: " + accuracy);

        for (String category : classifier.categories) {
            int tp = truePositive.getOrDefault(category, 0);
            int fp = falsePositive.getOrDefault(category, 0);
            int fn = falseNegative.getOrDefault(category, 0);

            double precision = tp + fp == 0 ? 0 : (double) tp / (tp + fp);
            double recall = tp + fn == 0 ? 0 : (double) tp / (tp + fn);
            double f1 = (precision + recall) == 0 ? 0 : 2 * (precision * recall) / (precision + recall);

            System.out.println("\nCategory: " + category);
            System.out.println("Precision: " + precision);
            System.out.println("Recall: " + recall);
            System.out.println("F1 Score: " + f1);
        }
    }
}
