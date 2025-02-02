import java.io.*;
import java.util.*;

public class AutonomousVehicleClassifier {

    // Cost constants
    private static final int COST_FOR_NOT_RETURNING = 8;
    private static final int COST_FOR_RETURNING_ON_TIME = 2;
    private static final int COST_FOR_RETURNING_UNNECESSARILY = 4;

    // Class probabilities
    private static final double PROB_DANGEROUS = 0.1;
    private static final double PROB_HARMLESS = 0.9;

    // Transition and initial probabilities
    private static Map<String, Map<String, Map<String, Double>>> transitionProbabilities; //Nested map to store state transition probabilities for G & H
    private static Map<String, Map<String, Double>> initialProbabilities; //Stores the probabilities of starting in a particular state for each G & H

    public static void main(String[] args) throws IOException {
        /*
        if (args.length < 2) {
            System.out.println("Usage: java -jar AutonomousVehicleClassifier.jar <train.txt> <eval.txt>");
            return;
        }
        String trainFile = args[0];
        String evalFile = args[1];
        */
        String trainFile = "src/train.txt";
        String evalFile = "src/eval.txt";

        trainModel(trainFile);
        printTransitionMatrices();
        int totalCost = evaluateModel(evalFile);
        System.out.println("Total Cost: " + totalCost);
    }

    /**
     * Maps a sensor value to a predefined state.
     */
    private static String mapToState(int value) {
        if ( value <= 27) {
            return "A";
        } else if (value >= 28 && value <= 42) {
            return "B";
        } else {
            return "C";
        }
    }

    /**
     * Trains the Markov model using the provided training data file.
     */
    private static void trainModel(String trainFile) throws IOException {
        transitionProbabilities = new HashMap<>();
        initialProbabilities = new HashMap<>();

        for (String label : new String[]{"G", "H"}) {
            transitionProbabilities.put(label, new HashMap<>());
            initialProbabilities.put(label, new HashMap<>());
        }

        BufferedReader br = new BufferedReader(new FileReader(trainFile));
        String line;

        while ((line = br.readLine()) != null) {
            String[] parts = line.split("\\s+");
            String label = parts[0];
            String[] sequence = Arrays.stream(parts, 1, parts.length)
                    .mapToInt(Integer::parseInt)
                    .mapToObj(AutonomousVehicleClassifier::mapToState)
                    .toArray(String[]::new);

            // Update initial probabilities
            initialProbabilities.get(label).merge(sequence[0], 1.0, Double::sum);//keep track of how often each state appears as the starting state for a given label.

            // Update transition probabilities
            for (int i = 0; i < sequence.length - 1; i++) {
                String current = sequence[i];
                String next = sequence[i + 1];

                transitionProbabilities.get(label)
                        .computeIfAbsent(current, k -> new HashMap<>())
                        .merge(next, 1.0, Double::sum);
            }
        }
        br.close();

        // Normalize probabilities
        normalizeProbabilities();
    }


     // Prints the transition matrices for the "G" and "H" labels.

     private static void printTransitionMatrices() {
     for (String label : new String[]{"G", "H"}) {
     System.out.println("Transition Matrix for Label: " + label);

     // Get the transitions for the current label
     Map<String, Map<String, Double>> transitions = transitionProbabilities.get(label);

     // Get all states (rows and columns of the matrix)
     Set<String> states = new HashSet<>();
     transitions.forEach((from, toMap) -> {
     states.add(from);
     states.addAll(toMap.keySet());
     });

     // Sort the states for consistent output
     List<String> sortedStates = new ArrayList<>(states);
     Collections.sort(sortedStates);

     // Print header row
     System.out.print("\t");
     for (String colState : sortedStates) {
     System.out.print(colState + "\t");
     }
     System.out.println();

     // Print rows with probabilities
     for (String rowState : sortedStates) {
     System.out.print(rowState + "\t");
     for (String colState : sortedStates) {
     double prob = transitions.getOrDefault(rowState, new HashMap<>()).getOrDefault(colState, 0.0);
     System.out.printf("%.2f\t", prob);
     }
     System.out.println();
     }
     System.out.println();
     }
     }


    /**
     * Normalizes initial and transition probabilities.
     */
    private static void normalizeProbabilities() {
        for (String label : new String[]{"G", "H"}) {
            // Normalize initial probabilities
            Map<String, Double> initial = initialProbabilities.get(label);
            double initialSum = initial.values().stream().mapToDouble(Double::doubleValue).sum();
            initial.replaceAll((k, v) -> v / initialSum);

            // Normalize transition probabilities
            Map<String, Map<String, Double>> transitions = transitionProbabilities.get(label);
            for (Map<String, Double> nextStateMap : transitions.values()) {
                double transitionSum = nextStateMap.values().stream().mapToDouble(Double::doubleValue).sum();
                nextStateMap.replaceAll((k, v) -> v / transitionSum);
            }
        }
    }

    /**
     * Predicts whether the sequence is "G" (dangerous) or "H" (harmless).
     */
    private static String predict(String[] sequence) {
        double probG = Math.log(PROB_DANGEROUS);
        double probH = Math.log(PROB_HARMLESS);

        // Include initial probability
        probG += Math.log(initialProbabilities.get("G").getOrDefault(sequence[0], 1e-2));
        probH += Math.log(initialProbabilities.get("H").getOrDefault(sequence[0], 1e-2));

        // Calculate transition probabilities
        for (int i = 0; i < sequence.length - 1; i++) {
            String current = sequence[i];
            String next = sequence[i + 1];

            probG += Math.log(transitionProbabilities.get("G")
                    .getOrDefault(current, new HashMap<>())
                    .getOrDefault(next, 1e-2));
            probH += Math.log(transitionProbabilities.get("H")
                    .getOrDefault(current, new HashMap<>())
                    .getOrDefault(next, 1e-2));
        }

        return (probG  > probH) ? "G" : "H";
    }

    /**
     * Evaluates the model using the evaluation file and computes the total cost.
     */
    private static int evaluateModel(String evalFile) throws IOException {
        int totalCost = 0;
        int correctPredictions = 0; // Counter for correct predictions
        int totalPredictions = 0;  // Total number of evaluations
        // Create a log to compare predictions and actual labels
        PrintWriter logWriter = new PrintWriter(new FileWriter("predictions_log.txt"));

        BufferedReader br = new BufferedReader(new FileReader(evalFile));
        String line;

        while ((line = br.readLine()) != null) {
            String[] parts = line.split("\\s+");
            String actualLabel = parts[0];
            String[] sequence = Arrays.stream(parts, 1, parts.length)
                    .mapToInt(Integer::parseInt)
                    .mapToObj(AutonomousVehicleClassifier::mapToState)
                    .toArray(String[]::new);

            String predictedLabel = predict(sequence);
            // Log predictions and actual labels
            logWriter.printf("Actual: %s, Predicted: %s%n", actualLabel, predictedLabel);

            // Count correct predictions
            if (actualLabel.equals(predictedLabel)) {
                correctPredictions++;
            }
            totalPredictions++;

            // Determine cost
            if (predictedLabel.equals("G") && actualLabel.equals("H")) {
                totalCost += COST_FOR_RETURNING_UNNECESSARILY;
            } else if (predictedLabel.equals("H") && actualLabel.equals("G")) {
                totalCost += COST_FOR_NOT_RETURNING;
            } else if (predictedLabel.equals("G") && actualLabel.equals("G")) {
                totalCost += COST_FOR_RETURNING_ON_TIME;
            }
        }
        br.close();
        logWriter.close();
        // Print summary of predictions
        System.out.println("Predictions log saved as predictions_log.txt"); //just to cross-check
        System.out.println("Predicted correctly " + correctPredictions + " times out of " + totalPredictions + " times.");
        double accuracy = (double) correctPredictions / totalPredictions * 100;
        System.out.printf("Accuracy: %.2f%%%n", accuracy);
        return totalCost;
    }

}

