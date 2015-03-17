import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.util.*;

/**
 * Created by vijay on 3/8/15.
 */
public class LR {
    /* Default values for learning parameters. Set in main */
    private static int vocabSize;
    private static double learnRate;
    private static double regularization;
    private static int maxPasses;
    private static int trainingSize;
    private static final String[] labelArray = {"nl","el","ru","sl","pl","ca","fr","tr","hu","de","hr","es","ga","pt"};
    private static double lambda = learnRate;
    private static double regularizationBase = 1.0 - 2*lambda*regularization;

    private static HashMap<String, Map<Integer, Integer>> A_PER_LABEL = new HashMap<String, Map<Integer, Integer>>();
    private static HashMap<String, Map<Integer, Double>> B_PER_LABEL = new HashMap<String, Map<Integer, Double>>();


    private static void initMaps() {
        HashMap<Integer, Integer> A;
        HashMap<Integer, Double> B;

        for (String label : labelArray) {
            A = new HashMap<Integer, Integer>();
            B = new HashMap<Integer, Double>();

            for (int i = 0; i < vocabSize; i++) {
                A.put(i, 0);
                B.put(i, 0.0);
            }

            A_PER_LABEL.put(label, A);
            B_PER_LABEL.put(label, B);
        }
    }

    /* Compute ID for each token. Generally unique */
    private static int getFeatureID(String token) {
        int id = token.hashCode() % vocabSize;
        return (id < 0) ? id + vocabSize : id;
    }

    /* Computes and returns a map of label -> p(label) */
    static double computeP(Vector<String> doc, Map<Integer, Double> B) {
        double dotProd = 0.0;
        for (String token : doc) {
            int id = getFeatureID(token);
            dotProd += B.get(id);
        }
        return Utils.sigmoid(dotProd);
    }


    // Cleanup final regularization for all labels
    private static void cleanupTrain(int k) {
        Map<Integer, Integer> A;
        Map<Integer, Double> B;

        for (String label : labelArray) {
            A = A_PER_LABEL.get(label);
            B = B_PER_LABEL.get(label);
            for (int j = 0; j < vocabSize; j++) {
                double regularizationFactor = Math.pow(regularizationBase, k - A.get(j));
                B.put(j, B.get(j) * regularizationFactor);
            }
        }
    }

    private static Set<Integer> getFeatureSet(Vector<String> tokens) {
        Set<Integer> featureIDs = new HashSet<Integer>();
        for (String token : tokens) {
            featureIDs.add(getFeatureID(token));
        }

        return featureIDs;
    }

    public static void train() throws Exception {
        int k = 0;
        int t = 0;
        double sum = 0.0;

        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String line;
        while ((line = br.readLine()) != null && line.length() != 0) {
            // Each line is of the form Cat1,Cat2,...,CatK  w1 w2 w3 ... wN
            String[] pair = line.split("\t");
            Set<String> labels = new HashSet<String>(Arrays.asList(pair[0].split(",")));
            Vector<String> tokens = Utils.tokenizeDoc(pair[1]);
            Set<Integer> tokenIDs = getFeatureSet(tokens);

            if (k % trainingSize == 0) {
                t++;
                lambda = learnRate/(t*t);
                regularizationBase = 1.0 - 2*lambda*regularization;
                //System.err.println("Iteration " + t);
                //System.err.println(sum);
                sum = 0.0;
            }
            k++;

            for (String label : labelArray) {
                Map<Integer, Integer> A = A_PER_LABEL.get(label);
                Map<Integer, Double> B = B_PER_LABEL.get(label);

                // Recompute p for the given label at the start of every example
                double p = computeP(tokens, B);

                // Does this label appear in the training labels?
                int y = (labels.contains(label)) ? 1 : 0;

                // Sum up log likelihoods
                sum += (y == 1) ? Math.log(p) : Math.log(1-p);

                for (int j = 0; j < vocabSize; j++) {
                    if (tokenIDs.contains(j)) {
                        double regularizationFactor = Math.pow(regularizationBase, k - A.get(j));
                        B.put(j, B.get(j) + lambda * (y - p));
                        B.put(j, B.get(j) * regularizationFactor);
                        A.put(j, k);
                    }
                }
            }
        }

        cleanupTrain(k);
        br.close();
        //System.err.println(sum);
    }

    private static double computeLabelScore(Vector<String> tokens, String label) {
        double result = 0.0;
        int id;
        Map<Integer, Double> B = B_PER_LABEL.get(label);
        for (String token : tokens) {
            id = getFeatureID(token);
            result += B.get(id);
        }
        return Utils.sigmoid(result);
    }

    public static void test(String testFilename) throws Exception {
        BufferedReader br = new BufferedReader(new FileReader(testFilename));
        String line;
        Vector<String> tokens;
        int correct = 0;
        int total = 0;

        while ((line = br.readLine()) != null) {
            String[] pair = line.split("\t");
            Set<String> labels = new HashSet<String>(Arrays.asList(pair[0].split(",")));
            tokens = Utils.tokenizeDoc(pair[1]);

            for (String label : labelArray) {
                double score = computeLabelScore(tokens, label);
                // Increment correct count if score correctly classifies label
                boolean truePositive = (score > 0.5) && labels.contains(label);
                boolean trueNegative = (score < 0.5) && !labels.contains(label);
                if (truePositive || trueNegative)
                    correct++;
                System.out.print(label + " " + score + ", ");
            }
            total += labelArray.length;

            System.out.println();
        }

        System.err.println(vocabSize + ": Accuracy: " + ((double) correct)/total);
        br.close();
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 6) {
            System.out.println("Invalid args.");
            return;
        }

        vocabSize = Integer.valueOf(args[0]);
        learnRate = Double.valueOf(args[1]);
        regularization = Double.valueOf(args[2]);
        maxPasses = Integer.valueOf(args[3]);
        trainingSize = Integer.valueOf(args[4]);


        String testFile = args[5];
        initMaps();
        train();
        test(testFile);

        /*for (String label : labelArray) {
            for (int i = 0; i < B_PER_LABEL.get(label).size(); i++) {
                System.out.println(label + " " + i + " " + B_PER_LABEL.get(label).get(i));
            }
        }*/
    }
}
