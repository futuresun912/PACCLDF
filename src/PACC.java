/**
 * Created by sunlu on 12/11/15.
 */
import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.cc.CNode;
import meka.core.A;
import meka.core.M;
import weka.core.Instances;
import weka.core.Instance;


public class PACC extends CC {

    // Training a PACC
    public void buildClassifier(Instances D) throws Exception {
        testCapabilities(D);
        int L = D.classIndex();

        // Learning of the polytree
        Polytree polytree = new Polytree();
        int[][] pa = polytree.polyTree(D, null);
        m_Chain = polytree.getChainOrder();

        if (getDebug()) {
            System.out.println(A.toString(m_Chain));
            System.out.println(M.toString(pa));
        }

        // Building the PACC
        nodes = new CNode[L];
        for (int j : m_Chain) {
            nodes[j] = new CNode(j, null, pa[j]);
            nodes[j].build(D, m_Classifier);
        }
    }

    // Test on a single instance deterministically
    public double[] distributionForInstance(Instance x) throws Exception {
        int L = x.classIndex();
        double[] y = new double[L];
        for (int j : m_Chain) {
            y[j] = nodes[j].classify(x, y);
        }
        return y;
    }

}


