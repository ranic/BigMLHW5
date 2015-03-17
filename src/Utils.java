import java.util.Vector;

/**
 * Created by vijay on 3/9/15.
 */
public class Utils {
    private static int overflow = 20;

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

    /* Computes the sigmoid function of the input */
    static double sigmoid(double x) {
        if (x > overflow)
            x = overflow;
        else if (x < -overflow)
            x = -overflow;

        return 1.0/(1.0 + Math.exp(-x));
    }
}
