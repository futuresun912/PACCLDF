/**
 * Created by sunlu on 12/9/15.
 */

import meka.classifiers.multilabel.*;

public class PerformMLC {

    static String n = "3";                       // n-fold CV
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

        BR mlClassifier0 = new BR();
        CC mlClassifier1 = new CC();
        BCC mlClassifier2 = new BCC();
        PACC mlClassifier3 = new PACC();
        BRLDF mlClassifier00 = new BRLDF();
        CCLDF mlClassifier10 = new CCLDF();
        BCCLDF mlClassifier20 = new BCCLDF();
        PACCLDF mlClassifier30 = new PACCLDF();

        String filename = arfflist[1];
        String mlMetond = "pacc";

//        switch (mlMetond) {
//            case "br":
//                setTestOptions(filename);
//                Evaluation.runExperiment(mlClassifier0, options);
//                break;
//            case "cc":
//                setTestOptions(filename);
//                Evaluation.runExperiment(mlClassifier1, options);
//                break;
//            case "bcc":
//                setTestOptions(filename);
//                Evaluation.runExperiment(mlClassifier2, options);
//                break;
//            case "pacc":
//                setTestOptions(filename);
//                Evaluation.runExperiment(mlClassifier3, options);
//                break;
//            case "brldf":
//                setTestOptions(filename);
//                Evaluation.runExperiment(mlClassifier00, options);
//                break;
//            case "ccldf":
//                setTestOptions(filename);
//                Evaluation.runExperiment(mlClassifier10, options);
//                break;
//            case "bccldf":
//                setTestOptions(filename);
//                Evaluation.runExperiment(mlClassifier20, options);
//                break;
//            case "paccldf":
//                setTestOptions(filename);
//                Evaluation.runExperiment(mlClassifier30, options);
//                break;
//        }


		 //**************************************************************
		 //************** Experiments on all methods ********************
		 //**************************************************************
		 for (int i = 9 ; i < 11; i ++) {
             System.out.println("*****************************************");
             System.out.println("data-"+i+" starts!");
             System.out.println("*****************************************\n");

             setTestOptions(arfflist[i]);
             EvaluationPro.runExperiment(mlClassifier3, options);

//             setTestOptions(arfflist[i]);
//             EvaluationPro.runExperiment(mlClassifier30, options);

             System.out.println("*****************************************");
             System.out.println("data-"+i+" is finished!");
             System.out.println("*****************************************\n");
		 }
	   //**************************************************************
	   //**************************************************************
	   //**************************************************************

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



