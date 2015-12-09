/**
 * Created by sunlu on 12/9/15.
 */
import meka.classifiers.multilabel.Evaluation;
import meka.classifiers.multilabel.MultilabelClassifier;
import meka.core.MLEvalUtils;
import weka.core.SerializationHelper;

import meka.core.MLUtils;
import meka.core.Result;
import weka.core.*;

import java.util.Random;


/**
 * Created by sunlu on 10/25/15.
 * For perform multiple experiments without stop.
 */
public class EvaluationPro extends Evaluation {

    public static void runExperiment(MultilabelClassifier h, String options[]) throws Exception {

        // Help
        if(Utils.getOptionPos('h',options) >= 0) {
            System.out.println("\nHelp requested");
            Evaluation.printOptions(h.listOptions());
            return;
        }

        h.setOptions(options);

        if (h.getDebug()) System.out.println("Loading and preparing dataset ...");

        // Load Instances from a file
        Instances D_train = loadDataset(options);

        // Try extract and set a class index from the @relation name
        MLUtils.prepareData(D_train);

        // Override the number of classes with command-line option (optional)
        if(Utils.getOptionPos('C',options) >= 0) {
            int L = Integer.parseInt(Utils.getOption('C',options));
            D_train.setClassIndex(L);
        }

        // We we still haven't found -C option, we can't continue (don't know how many labels)
        int L = D_train.classIndex();
        if(L <= 0) {
            throw new Exception("[Error] Number of labels not specified.\n\tYou must set the number of labels with the -C option, either inside the @relation tag of the Instances file, or on the command line.");
            // apparently the dataset didn't contain the '-C' flag, check in the command line options ...
        }


        // Randomize (Instances)
        int seed = (Utils.getOptionPos('s',options) >= 0) ? Integer.parseInt(Utils.getOption('s',options)) : 0;
        if(Utils.getFlag('R',options)) {
            D_train.randomize(new Random(seed));
        }

        // Randomize (Model) DEPRECATED
        if (h instanceof Randomizable) {
			/*
			   THIS WILL BE DEPRECATED, METHODS SHOULD IMPLEMENT THEIR OWN OPTION FOR THE SEED (Or, maybe, just override this one)
			   As it will be commented out, results could be a bit different (but not significantly so)
			   */
            //int method_seed = (Utils.getOptionPos('S',options) >= 0) ? Integer.parseInt(Utils.getOption('S',options)) : 1;
            //System.out.println("set seed = "+method_seed);
            ((Randomizable)h).setSeed(seed);
        }

        // Verbosity Option
        String voption = "1";
        if (Utils.getOptionPos("verbosity",options) >= 0) {
            voption = Utils.getOption("verbosity",options);
        }

        // Save for later?
        //String fname = null;
        //if (Utils.getOptionPos('f',options) >= 0) {
        //	fname = Utils.getOption('f',options);
        //}
        // Dump for later?
        String dname = null;
        if (Utils.getOptionPos('d',options) >= 0) {
            dname = Utils.getOption('d',options);
        }
        // Load from file?
        String lname = null;
        if (Utils.getOptionPos('l',options) >= 0) {
            lname = Utils.getOption('l',options);
            h = (MultilabelClassifier)SerializationHelper.read(lname);
            //Object o[] = SerializationHelper.readAll(lname);
            //h = (MultilabelClassifier)o[0];
        }

        try {

            Result r = null;

            // Threshold OPtion
            String top = "PCut1"; // default
            if (Utils.getOptionPos("threshold",options) >= 0)
                top = Utils.getOption("threshold",options);

            if(Utils.getOptionPos('x',options) >= 0) {
                // CROSS-FOLD-VALIDATION

                int numFolds = MLUtils.getIntegerOption(Utils.getOption('x',options),10); // default 10
                // Check for remaining options
                Utils.checkForRemainingOptions(options);
                Result fold[] = Evaluation.cvModel(h,D_train,numFolds,top,voption);
                r = MLEvalUtils.averageResults(fold);
                System.out.println(r.toString());
                //if (fname != null) {
                //for(int i = 0; i < fold.length; i++) {
                //Result.writeResultToFile(fold[i],fname+"."+i);
                //}
                //}
            }
            else {
                // TRAIN-TEST SPLIT

                Instances D_test = null;

                if(Utils.getOptionPos('T',options) >= 0) {
                    // load separate test set
                    try {
                        D_test = loadDataset(options,'T');
                        MLUtils.prepareData(D_test);
                    } catch(Exception e) {
                        throw new Exception("[Error] Failed to Load Test Instances from file.", e);
                    }
                }
                else {
                    // split training set into train and test sets
                    // default split
                    int N_T = (int)(D_train.numInstances() * 0.60);
                    if(Utils.getOptionPos("split-percentage",options) >= 0) {
                        // split by percentage
                        double percentTrain = Double.parseDouble(Utils.getOption("split-percentage",options));
                        N_T = (int)Math.round((D_train.numInstances() * (percentTrain/100.0)));
                    }
                    else if(Utils.getOptionPos("split-number",options) >= 0) {
                        // split by number
                        N_T = Integer.parseInt(Utils.getOption("split-number",options));
                    }

                    int N_t = D_train.numInstances() - N_T;
                    D_test = new Instances(D_train,N_T,N_t);
                    D_train = new Instances(D_train,0,N_T);
                }

                // Invert the split?
                if(Utils.getFlag('i',options)) { //boolean INVERT 			= Utils.getFlag('i',options);
                    Instances temp = D_test;
                    D_test = D_train;
                    D_train = temp;
                }

                // Check for remaining options
                Utils.checkForRemainingOptions(options);

                if (h.getDebug()) System.out.println(":- Dataset -: "+MLUtils.getDatasetName(D_train)+"\tL="+L+"\tD(t:T)=("+D_train.numInstances()+":"+D_test.numInstances()+")\tLC(t:T)="+Utils.roundDouble(MLUtils.labelCardinality(D_train,L),2)+":"+Utils.roundDouble(MLUtils.labelCardinality(D_test,L),2)+")");

                if (lname != null) {
                    // h is already built, and loaded from a file, test it!
                    r = testClassifier(h,D_test);
                    // @note: unfortunately, we have to do this again -- but with a threshold
                    //        we cannnot just call
                    // 				r = evaluateModel(h,test,top,voption);
                    // because of the threshold -- which (if we use PCut1 or PCutL)
                    // we calculate from training data + predictions! like this:
                    String t = MLEvalUtils.getThreshold(r.predictions, D_train, top);
                    // so then, we recalculate the statistics with that threshold:
                    r = evaluateModel(h,D_test,t,voption);
                }
                else {
                    r = evaluateModel(h,D_train,D_test,top,voption);
                }
                // @todo, if D_train==null, assume h is already trained
                System.out.println(r.toString());
            }

            // Save ranking data to file?
            //if (fname != null) {
            //	Result.writeResultToFile(r,fname);
            //}
            // Save model to file?
            if (dname != null) {
                SerializationHelper.write(dname, (Object) h);
            }

        } catch(Exception e) {
            e.printStackTrace();
            Evaluation.printOptions(h.listOptions());
            System.exit(1);
        }

        System.gc();
//        System.exit(0);
    }

}

