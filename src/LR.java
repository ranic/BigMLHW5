import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.*;

/**
 * Created by vijay on 3/8/15.
 */
public class LR {
    /* Default values for learning parameters */
    private static int vocabSize = 10000;
    private static double learnRate = 0.5;
    private static double regularization = 0.1;
    private static int maxPasses = 20;
    private static int trainingSize = 1000;

    private static final String[] labelArray = {"nl","el","ru","sl","pl","ca","fr","tr","hu","de","hr","es","ga","pt"};
    private static final Set<String> labelFilter = new HashSet<String>(Arrays.asList(labelArray));

    private static HashMap<String, Map<Integer, Double>> params = new HashMap<String, Map<Integer, Double>>();

    /* Compute ID for each token. Generally unique */
    private static int computeTokenID(String token) {
        int id = token.hashCode() % vocabSize;
        return (id < 0) ? id + vocabSize : id;
    }


    static Vector<String> tokenizeDoc(String cur_doc) {
        String[] words = cur_doc.toLowerCase().split("\\s+");
        Vector<String> tokens = new Vector<String>();
        for (int i = 0; i < words.length; i++) {
            words[i] = words[i].replaceAll("\\W", "");
            if (words[i].matches("[a-zA-z]{3,}")) {
                tokens.add(words[i]);
            }
        }
        return tokens;
    }

    static Set<String> filterLabels(String[] labels) {
        if (labelFilter.isEmpty()) {
            return new HashSet<String>(Arrays.asList(labels));
        } else {
            Set<String> result = new HashSet<String>();
            for (String s : labels) {
                if (labelFilter.contains(s)) {
                    result.add(s);
                }
            }
            return result;
        }
    }

    /* Computes the sigmoid function of the input */
    static double sigmoid(double x) {
        return 1.0/(1.0 + Math.exp(-x));
    }


    /* Computes and returns a map of label -> p(label) */
    static double computeP(Vector<String> doc, Map<Integer, Double> B) {
        double dotProd = 0.0;
        for (String token : doc) {
            int id = computeTokenID(token);
            if (B.containsKey(id)) {
                dotProd += B.get(id);
            }
        }
        return sigmoid(dotProd);
    }



    public static Map<Integer, Double> RegularizedLR(String label) {
        double p;
        HashMap<Integer, Integer> A = new HashMap<Integer, Integer>();
        HashMap<Integer, Double> B = new HashMap<Integer, Double>();
        for (int j = 0; j < vocabSize; j++) {
            A.put(j, 0);
            B.put(j, 0.0);
        }
        int id;
        int k = 0;
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
            String line;
            while ((line = br.readLine()) != null && line.length() != 0) {
                // Each line is of the form Cat1,Cat2,...,CatK  w1 w2 w3 ... wN
                String[] pair = line.split("\t");
                Set<String> labels = filterLabels(pair[0].split(","));
                Vector<String> tokens = tokenizeDoc(pair[1]);
                HashMap<Integer, Integer> tokenIDs = new HashMap<Integer, Integer>();
                k++;

                // Build a table of ID -> count for tokens occurring in doc
                for (String token : tokens) {
                    id = computeTokenID(token);
                    if (!tokenIDs.containsKey(id))
                        tokenIDs.put(id, 0);

                    tokenIDs.put(id, tokenIDs.get(id) + 1);
                }

                // Recompute p for the given label at the start of every example
                p = computeP(tokens, B);

                // Does this label appear in the training labels?
                int y = (labels.contains(label)) ? 1 : 0;

                for (int j = 0; j < vocabSize; j++) {
                    if (tokenIDs.containsKey(j)) {
                        double regularizationFactor = Math.pow(1.0 - 2 * learnRate * regularization, k - A.get(j));
                        for (int i = 0; i < tokenIDs.get(j); i++) {
                            B.put(j, B.get(j) + learnRate * (y - p));
                            B.put(j, B.get(j) * regularizationFactor);
                        }
                        A.put(j, k);
                    }
                }
            }

        } catch (Exception e) {
            System.err.println("Error:" + e.getMessage());
            e.printStackTrace();
        }

        for (int j = 0; j < vocabSize; j++) {
            double regularizationFactor = Math.pow(1.0 - 2 * learnRate * regularization, k - A.get(j));
            B.put(j, B.get(j) * regularizationFactor);
        }

        return B;
    }


    /*public static void LR() {
        try {
            HashMap<String, Double> p;
            HashMap<Integer, Double> B;
            double pLabel;
            int id;
            BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

            String line;
            while ((line = br.readLine()) != null && line.length() != 0) {
                // Each line is of the form Cat1,Cat2,...,CatK  w1 w2 w3 ... wN
                String[] pair = line.split("\t");
                Set<String> labels = filterLabels(pair[0].split(","));
                Vector<String> tokens = tokenizeDoc(pair[1]);

                // Recompute p for all labels at the start of every example
                p = computeP(tokens);

                // Each label is a separate classifier, so treat all separately
                for (String label : labelFilterArray) {
                    B = params.get(label);
                    pLabel = p.get(label);
                    int y = (labels.contains(label)) ? 1 : 0;

                    for (String token : tokens) {
                        id = computeTokenID(token);
                        if (!B.containsKey(id)) {
                            B.put(id, 0.0);
                        }

                        B.put(id, B.get(id) + learnRate * (y - pLabel));
                    }
                }
            }

        } catch (Exception e) {
            System.err.println("Error:" + e.getMessage());
        }

    }*/



    public static void main(String[] args) {
        /*if (args.length != 5) {
            System.out.println("Invalid args.");
            return;
        }

        vocabSize = Integer.valueOf(args[0]);
        learnRate = Double.valueOf(args[1]);
        regularization = Double.valueOf(args[2]);
        maxPasses = Integer.valueOf(args[3]);
        trainingSize = Integer.valueOf(args[4]);
*/
        for (String label : labelArray)
            params.put(label, RegularizedLR(label));

        for (String label : labelArray) {
            for (int i = 0; i < params.get(label).size(); i++) {
                System.out.println(label + " " + i + " " + params.get(label).get(i));
            }
        }
    }
}
