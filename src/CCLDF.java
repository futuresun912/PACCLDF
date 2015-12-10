/**
 * Created by sunlu on 12/10/15.
 */

import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.cc.CNode;
import meka.core.A;
import weka.core.Instance;
import weka.core.Instances;
import java.util.Random;


public class CCLDF extends CC {

    private MLFeaSelect mlFeaSelect;

    public void buildClassifier(Instances D) throws Exception {
        testCapabilities(D);
        int L = D.classIndex();

        // Get the IR factor for Wrapper
        double[] IRfactor = StatUtilsPro.CalcIRFactor(D);

        // First-stage feature selection
        double perFea = getPerFeature(D);
        mlFeaSelect = new MLFeaSelect(L);
        mlFeaSelect.setPercentFeature(perFea);
        mlFeaSelect.feaSelect1(D, L);
//        mlFeaSelect.feaSelect1IR(D, L, IRfactor);

        m_R = new Random(m_S);
        int[] indices = getChain();
        if (indices == null) {
            indices = A.make_sequence(L);
            A.shuffle(indices, m_R);
            setChain(indices);
        }

        nodes = new CNode[L];
        int[] pa = new int[]{};
        for (int j : m_Chain) {
            Instances tempD = mlFeaSelect.instTransform(D, j);
            mlFeaSelect.feaSelect2(tempD, j);
//            mlFeaSelect.feaSelect2PA(tempD, j);
            tempD = mlFeaSelect.instTransform(D, j);
            nodes[j] = new CNode(j, null, pa);
            nodes[j].build(tempD, m_Classifier);
            pa = A.append(pa, j);

        }

    }

    public double[] distributionForInstance(Instance x) throws Exception {
        int L = x.classIndex();
        double[] y = new double[L];
        for (int j : m_Chain) {
            Instance xd = mlFeaSelect.instTransform(x, j);
            y[j] = nodes[j].classify(xd, y);
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
