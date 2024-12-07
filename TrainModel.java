import weka.classifiers.Classifier;
import weka.classifiers.trees.J48; // Example classifier
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.SerializationHelper;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class TrainModel {

    public static void main(String[] args) {
        try {
            // Load dataset
            String csvFile = "Covid_dataset.csv";
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(csvFile));
            Instances data = loader.getDataSet();
            data.setClassIndex(data.numAttributes() - 1); // Set class index

            // Check class distribution
            System.out.println("Class Distribution: " + data.attributeStats(data.classIndex()).nominalCounts);

            // Train classifier
            Classifier classifier = new J48(); // Example: J48 decision tree
            classifier.buildClassifier(data);

            // Save model
            SerializationHelper.write("Covid_model3.model", classifier);
            System.out.println("Model trained and saved successfully.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
