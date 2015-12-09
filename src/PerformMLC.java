/**
 * Created by sunlu on 12/9/15.
 */

import meka.classifiers.multilabel.*;

public class PerformMLC {

    static String n = "5";                       // n-fold CV
    static String percent = "75.0";              // split percentage
    static String outputType = "2";              // 1, 2, 3, 4, 5, 6
    static String baseline = "Logistic";         // Logistic, NaiveBayes, SMO
    static boolean flagCV = true;                // use CV if true
    static boolean flagDebug = false;            // output debug infor. if true
    static String[] options = new String[12];    // specified options
    static String arfflist[] = {
            "scene",        // 0
            "emotions",     // 1
            "flags",        // 2
            "yeast",        // 3
            "birds",        // 4
            "genbase",      // 5
            "medical",      // 6
            "enron",        // 7
            "languagelog",  // 8
            "bibtex",       // 9
            "Corel5k",      // 10
            "mediamill",    // 11
            "CAL500",       // 12
            "rcv1subset1",  // 13
            "rcv1subset3",  // 14
            "delicious"     // 15
    };

    public static void  main(String[] args) throws Exception {

        BR testClassifier0 = new BR();
        CC testClassifier1 = new CC();

        String filename = arfflist[1];

        int i = 0;

        switch (i) {
            case 0:
                setTestOptions(filename);
                Evaluation.runExperiment(testClassifier0, options);
                break;
            case 1:
                setTestOptions(filename);
                Evaluation.runExperiment(testClassifier1, options);
                break;

        }


//		 //**************************************************************
//		 //************** Experiments on all methods ********************
//		 //**************************************************************
//		 for (int i = 0 ; i < arfflist.length; i ++) { // traverse all data sets
//             System.out.println("*****************************************");
//             System.out.println("data-"+i+" starts!");
//             System.out.println("*****************************************\n");
////			 setTestOptions(arfflist[i], "br", 1, 0);
////			 EvaluationPro.runExperiment(testClassifier0, options);
////
//////			 setTestOptions(arfflist[i], "bcc", 1, 0);
//////			 EvaluationPro.runExperiment(testClassifier2, options);
////
////             setTestOptions(arfflist[i], "bccpro", 1, 0);
////             EvaluationPro.runExperiment(testClassifier21, options);
//             System.out.println("*****************************************");
//             System.out.println("data-"+i+" is finished!");
//             System.out.println("*****************************************\n");
//		 }
//	   //**************************************************************
//	   //**************************************************************
//	   //**************************************************************

    }

    public static void setTestOptions(String arffname) {
        // select a data set
        options[0] = "-t";
        options[1] = "/home/sunlu/workspace/data/" + arffname + ".arff";

        if (flagCV) {
            // use n-fold cross validation
            options[2] = "-x";
            options[3] = n;
        } else {
            // split train/test in percent%
            options[2] = "-split-percentage";
            options[3] = percent;
        }

        // seed for randomizing
        options[4] = "-s";
        options[5] = "1";
        options[6] = "-R";

        // output type
        options[7] = "-verbosity";
        options[8] = outputType;

        // choose the baseline classifier
        options[9] = "-W";
        if (baseline == "NaiveBayes")
            options[10] = "weka.classifiers.bayes.NaiveBayes";
        else
            options[10] = "weka.classifiers.functions." + baseline;

        // output debug information
        if (flagDebug)
            options[11] = "-output-debug-info";
        else
            options[11] = "";
    }
}



