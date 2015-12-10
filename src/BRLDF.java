/**
 * Created by sunlu on 12/10/15.
 */

import meka.classifiers.multilabel.BR;
import meka.core.MLUtils;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class BRLDF extends BR {

    protected Classifier m_MultiClassifiers[] = null;
    protected Instances m_InstancesTemplates[] = null;
    protected MLFeaSelect mlFeaSelect;

    public void buildClassifier(Instances D) throws Exception {
        testCapabilities(D);
        int L = D.classIndex();

        // Get the Imbalance ratio-related statistics
//        double[] IRfactor = StatUtilsPro.CalcIRFactor(D);

        m_MultiClassifiers = AbstractClassifier.makeCopies(m_Classifier, L);
        m_InstancesTemplates = new Instances[L];

        // First-stage feature selection
        double perFea = getPerFeature(D);
        mlFeaSelect = new MLFeaSelect(L);
        mlFeaSelect.setPercentFeature(perFea);
        mlFeaSelect.feaSelect1(D, L);
//        mlFeaSelect.feaSelect1IR(D, L, IRfactor);

        for(int j = 0; j < L; j++) {

            Instances tempD = mlFeaSelect.instTransform(D, j);
            mlFeaSelect.feaSelect2(tempD, j);
//            mlFeaSelect.feaSelect2PA(tempD, j);
            tempD = mlFeaSelect.instTransform(D, j);

            // Remove labels except j-th
            Instances D_j = MLUtils.keepAttributesAt(new Instances(tempD), new int[]{j}, L);
            D_j.setClassIndex(0);

            //Build the classifier for that class
            m_MultiClassifiers[j].buildClassifier(D_j);
            m_InstancesTemplates[j] = new Instances(D_j, 0);
        }
    }

    @Override
    public double[] distributionForInstance(Instance x) throws Exception {
        int L = x.classIndex();
        double y[] = new double[L];
        for (int j = 0; j < L; j++) {
            Instance xd = mlFeaSelect.instTransform(x, j);
            xd.setDataset(null);
            xd = MLUtils.keepAttributesAt((Instance)xd.copy(), new int[]{j}, L);
            xd.setDataset(m_InstancesTemplates[j]);
            y[j] = m_MultiClassifiers[j].distributionForInstance(xd)[1];
        }
        return y;
    }

    // estimate the number of selected features
    protected double getPerFeature(Instances D) throws Exception {
        int L = D.classIndex();
        int d = D.numAttributes() - L;
        double perTemp;
        if (d < 500) {
            perTemp = 0.4;
        } else if ( d < 1000 ) {
            perTemp = 0.2;
        } else if ( d < 1500 ) {
            perTemp = 0.1;
        } else if ( d < 2000 ){
            perTemp = 0.05;
        } else {
            perTemp = 0.01;
        }
        mlFeaSelect = new MLFeaSelect(L);
        mlFeaSelect.setPercentFeature(perTemp);
        mlFeaSelect.feaSelect1(D, 1);
        Instances tempD = mlFeaSelect.instTransform(D, 0);
        mlFeaSelect.feaSelect2(tempD, 0);
        int num = mlFeaSelect.getNumFeaCfs(0);
        return (double)num*2.0 / d;
    }

}

