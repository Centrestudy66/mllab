import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;
import weka.core.converters.CSVLoader;
import weka.core.Instances;
import weka.core.Utils;

import java.io.File;
import java.util.Random;

public class NaiveBayesTextClassifier2 {
    public static void main(String[] args) throws Exception {
        // Load dataset from CSV file
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("dataset.csv")); // Replace with your actual file path
        Instances data = loader.getDataSet();

        // Set the class attribute (Category column is the last one)
        data.setClassIndex(data.numAttributes() - 1);

        // Train Naïve Bayes classifier
        NaiveBayes nbClassifier = new NaiveBayes();
        nbClassifier.buildClassifier(data);

        // Perform 10-fold cross-validation
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(nbClassifier, data, 10, new Random(1));

        // Print evaluation metrics
        System.out.println("Accuracy: " + (1 - eval.errorRate()));
        System.out.println("Precision: " + eval.weightedPrecision());
        System.out.println("Recall: " + eval.weightedRecall());
        System.out.println(eval.toSummaryString());
    }
}
