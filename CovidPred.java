import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import weka.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.SerializationHelper;

class CovidPredictionAppGUI {

    static Classifier classifier;
    static Instances dataset;
    static JFrame frame;
    static JTextArea resultArea;
    static JRadioButton[] yesButtons;
    static JRadioButton[] noButtons;

    public static void main(String[] args) {
        // Load dataset and model
        loadDataset("Covid_dataset.csv");
        loadModel("Covid_model3.model");

        // Create and set up the window
        frame = new JFrame("COVID-19 Symptom Checker");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(400, 300);

        // Create the panel for input
        JPanel panel = new JPanel();
        panel.setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.insets = new Insets(10, 10, 10, 10);

        String[] labels = { "Breathing Problem", "Fever", "Dry Cough", "Sore Throat", "Running Nose" };

        yesButtons = new JRadioButton[labels.length];
        noButtons = new JRadioButton[labels.length];
        ButtonGroup[] buttonGroups = new ButtonGroup[labels.length];

        // Add input fields to the panel
        for (int i = 0; i < labels.length; i++) {
            gbc.gridx = 0;
            gbc.gridy = i;
            JLabel label = new JLabel(labels[i] + ":");
            panel.add(label, gbc);

            gbc.gridx = 1;
            yesButtons[i] = new JRadioButton("Yes");
            noButtons[i] = new JRadioButton("No");
            buttonGroups[i] = new ButtonGroup();
            buttonGroups[i].add(yesButtons[i]);
            buttonGroups[i].add(noButtons[i]);
            panel.add(yesButtons[i], gbc);

            gbc.gridx = 2;
            panel.add(noButtons[i], gbc);
        }

        // Add the result area
        resultArea = new JTextArea(5, 30);
        resultArea.setLineWrap(true);
        resultArea.setWrapStyleWord(true);
        resultArea.setEditable(false);
        JScrollPane scrollPane = new JScrollPane(resultArea);
        gbc.gridx = 0;
        gbc.gridy = labels.length;
        gbc.gridwidth = 3;
        panel.add(scrollPane, gbc);

        // Add submit button
        JButton submitButton = new JButton("Check Symptoms");
        gbc.gridx = 0;
        gbc.gridy = labels.length + 1;
        gbc.gridwidth = 3;
        submitButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                double[] features = new double[labels.length];

                for (int i = 0; i < labels.length; i++) {
                    features[i] = yesButtons[i].isSelected() ? 1 : 0;
                }

                Instance instance = createInstance(features);
                String result = predictCovidStatus(instance);
                resultArea.setText(result);
            }
        });
        panel.add(submitButton, gbc);

        // Add the panel to the frame
        frame.add(panel);
        frame.setVisible(true);
    }

    public static void loadDataset(String csvFile) {
        try {
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(csvFile));
            dataset = loader.getDataSet();
            dataset.setClassIndex(dataset.numAttributes() - 1);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void loadModel(String modelFile) {
        try {
            classifier = (Classifier) SerializationHelper.read(modelFile);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static Instance createInstance(double[] features) {
        Instance templateInstance = dataset.firstInstance();
        Instance newInstance = new DenseInstance(templateInstance);
        newInstance.setDataset(dataset);

        for (int i = 0; i < features.length; i++) {
            newInstance.setValue(dataset.attribute(i), features[i]);
        }
        return newInstance;
    }

    public static String predictCovidStatus(Instance instance) {
        try {
            double result = classifier.classifyInstance(instance);
            return result == 0 ? "Prediction: Low Risk of COVID-19. Stay Safe and Monitor Symptoms."
                    : "Prediction: High Risk of COVID-19. Please seek medical advice.";
        } catch (Exception e) {
            e.printStackTrace();
            return "Error in prediction.";
        }
    }
}
