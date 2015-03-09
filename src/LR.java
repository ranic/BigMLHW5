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

    private static final String[] labelFilterArray = {"nl","el","ru","sl","pl","ca","fr","tr","hu","de","hr","es","ga","pt"};
    private static final Set<String> labelFilter = new HashSet<String>(Arrays.asList(labelFilterArray));

    /* Map from label to a map from word position to a parameter weight for that position */
    private static HashMap<String, HashMap<Integer, Double>> params = new HashMap<String, HashMap<Integer, Double>>();


    /* Create a parameter HashMap for each label */
    private static void initParams() {
        for (String label : labelFilterArray) {
            params.put(label, new HashMap<Integer, Double>());
        }
    }

    /* Compute ID for each token. Generally unique */
    private static int computeTokenID(String token) {
        int id = token.hashCode() % vocabSize;
        return (id < 0) ? id + vocabSize : id;
    }


    /**
     * @param cur_doc
     * @return
     *  Vector of tokens from cur_doc
     */
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
    static HashMap<String, Double> computeP(Vector<String> doc) {
        HashMap<String, Double> result = new HashMap<String, Double>();
        HashMap<Integer, Double> B;
        for (String label : labelFilterArray) {
            double dotProd = 0.0;
            B = params.get(label);
            for (String token : doc) {
                int id = computeTokenID(token);
                if (B.containsKey(id)) {
                    dotProd += B.get(id);
                }
            }

            result.put(label, sigmoid(dotProd));
        }
        return result;
    }


    public static void LR() {
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

    }



    public static void main(String[] args) {
        if (args.length != 5) {
            System.out.println("Invalid args.");
            return;
        }

        vocabSize = Integer.valueOf(args[0]);
        learnRate = Double.valueOf(args[1]);
        regularization = Double.valueOf(args[2]);
        maxPasses = Integer.valueOf(args[3]);
        trainingSize = Integer.valueOf(args[4]);

        LR();
    }
}
