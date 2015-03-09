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
    }
}
