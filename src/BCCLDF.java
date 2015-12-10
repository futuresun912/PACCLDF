/**
 * Created by sunlu on 12/10/15.
 * 1. Marginal dependence modeled by normalized mutual information;
 * 2. max-sum algorithm for prediction
 */

import meka.classifiers.multilabel.BCC;
import meka.classifiers.multilabel.cc.CNode;
import meka.core.A;
import meka.core.M;
import mst.Edge;
import mst.EdgeWeightedGraph;
import mst.KruskalMST;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.*;

public class BCCLDF extends BCC {

    private MLFeaSelect mlFeaSelect;
    private int[][] pa;
    private int[][] ch;

    public void buildClassifier(Instances D) throws Exception {
        testCapabilities(D);
        m_R = new Random(getSeed());
        int L = D.classIndex();
        int d = D.numAttributes()-L;
//        double[] IRfactor = StatUtilsPro.CalcIRFactor(D);

        // CD is the normalized mutual information matrix.
        double[][] CD = StatUtilsPro.NormMargDep(D);

        if (getDebug())
            System.out.println("Normalized MI matrix: \n" + M.toString(CD));

        CD = M.multiply(CD,-1); // because we want a *maximum* spanning tree
        if (getDebug())
            System.out.println("Make a graph ...");
        EdgeWeightedGraph G = new EdgeWeightedGraph((int)L);
        for(int i = 0; i < L; i++) {
            for(int j = i+1; j < L; j++) {
                Edge e = new Edge(i, j, CD[i][j]);
                G.addEdge(e);
            }
        }

		/*
		 * Run an off-the-shelf MST algorithm to get a MST.
		 */
        if (getDebug())
            System.out.println("Get an MST ...");
        KruskalMST mst = new KruskalMST(G);

		/*
		 * Define graph connections based on the MST.
		 */
        int paM[][] = new int[L][L];
        for (Edge e : mst.edges()) {
            int j = e.either();
            int k = e.other(j);
            paM[j][k] = 1;
            paM[k][j] = 1;
            //StdOut.println(e);
        }
        if (getDebug()) System.out.println(M.toString(paM));

		/*
		 *  Turn the DAG into a Tree with the m_Seed-th node as root
		 */
        int root = getSeed();
        if (getDebug())
            System.out.println("Make a Tree from Root "+root);
        pa = new int[L][0];
        int visted[] = new int[L];
        Arrays.fill(visted,-1);
        visted[root] = 0;
        treeify(root,paM,pa, visted);
        if (getDebug()) {
            for(int i = 0; i < L; i++) {
                System.out.println("pa_"+i+" = "+Arrays.toString(pa[i]));
            }
        }

        // obtain the children for each node
        ch = new int[L][];
        for (int j = 0; j < L; j ++) {
            ch[j] = new int[]{};
            for (int k = 0; k < L; k ++) {
                for (int l : pa[k]) {
                    if ( l == j )
                        ch[j] = A.append(ch[j], k);
                }
            }
        }

        m_Chain = Utils.sort(visted);
        if (getDebug())
            System.out.println("sequence: "+Arrays.toString(m_Chain));
	   /*
		* Bulid a classifier 'tree' based on the Tree
		*/
        if (getDebug()) System.out.println("Build Classifier Tree ...");
        nodes = new CNode[L];

        // First-stage feature selection
        double perFea = getPerFeature(D);
        mlFeaSelect = new MLFeaSelect(L);
        mlFeaSelect.setPercentFeature(perFea);
        mlFeaSelect.feaSelect1(D, L);
//        mlFeaSelect.feaSelect1IR(D, L, IRfactor);

        for(int j : m_Chain) {
            if (getDebug())
                System.out.println("\t node h_"+j+" : P(y_"+j+" | x_[1:"+d+"], y_"+Arrays.toString(pa[j])+")");
            Instances tempD = mlFeaSelect.instTransform(D, j);
            mlFeaSelect.feaSelect2(tempD, j);
//            mlFeaSelect.feaSelect2PA(tempD, j);
            tempD = mlFeaSelect.instTransform(D, j);
            nodes[j] = new CNode(j, null, pa[j]);
            nodes[j].build(tempD, m_Classifier);

        }

        if (getDebug()) System.out.println(" * DONE * ");

    }

    /**
     * Treeify - make a tree given the structure defined in paM[][], using the root-th node as root.
     */
    private void treeify(int root, int paM[][], int paL[][], int visited[]) {
        int children[] = new int[]{};
        for(int j = 0; j < paM[root].length; j++) {
            if (paM[root][j] == 1) {
                if (visited[j] < 0) {
                    children = A.append(children, j);
                    paL[j] = A.append(paL[j],root);
                    visited[j] = visited[Utils.maxIndex(visited)] + 1;
                }
            }
        }
        // go through again
        for(int child : children) {
            treeify(child,paM,paL,visited);
        }
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

//    public double[] distributionForInstance(Instance x) throws Exception {
//        int L = x.classIndex();
//        double[] y = new double[L];
//        for (int j : m_Chain) {
//            Instance xd = mlFeaSelect.instTransform(x, j);
//            y[j] = nodes[j].classify(xd, y);
//        }
//        return y;
//    }

    // ***************************************************************************************
    // THE MAX-SUM ALGORITHM FOR PREDICTION **************************************************
    // ***************************************************************************************
    @Override
    public double[] distributionForInstance(Instance xy) throws Exception {

        int L = xy.classIndex();
        double y[];                            // save the optimal assignment
        double[][] yMax = new double[L][];     // save the y_j for local maximum
        double msgSum[][];                     // save the sum of log probability for y_j = 0 and 1
        double y_[];                           // traverse the path on parent nodes in push function
        double[][] msg = new double[L][];      // the message passing upwards (ragged array)
        double[][][] cpt = new double[L][2][]; // conditional probability tables for all nodes
        int paL;                               // save the number of parents of current node
        int powNumJ;                           // the number of 2 to the power of pa[j].length

        // Step 1: calculate the CPT for all nodes (from root(s) to leaf(s))
        for (int j : m_Chain ) {
            y = new double[L];
            paL = pa[j].length;
            powNumJ = (int) Math.pow(2, paL);
            cpt[j][0] = new double[powNumJ];
            cpt[j][1] = new double[powNumJ];
            y_ = new double [paL];
            Instance xd = mlFeaSelect.instTransform(xy, j);

            for (int k = 0; k < powNumJ; k ++) {
                for (int m = 0; m < paL; m ++) {
                    y[pa[j][m]] = y_[m];
                }
                cpt[j][0][k] = nodes[j].distribution(xd,y)[0];
                cpt[j][1][k] = 1.0 - cpt[j][0][k];
                if( push(y_, paL-1) ) {
                    break;
                }
            }
        }

        // Step 2: receive all the messages sent by the children of j  (from leaf(s) to root(s))
        for ( int i = L-1; i >= 0; i--)  {
            int j = m_Chain[i];
            paL = pa[j].length;
            powNumJ = (int) Math.pow(2, paL);
            msg[j] = new double[powNumJ];
            msgSum = new double[2][powNumJ];
            yMax[j] = new double[powNumJ];

            // initialization of msgSum j's CPT
            for (int k = 0; k < powNumJ; k ++) {
                msgSum[0][k] = Math.log(cpt[j][0][k]);
                msgSum[1][k] = Math.log(cpt[j][1][k]);
            }
            // collect msg from j's children into msgSum
            for ( int c : ch[j] ) {
                for ( int k = 0; k < powNumJ; k ++ ) {
                    msgSum[0][k] += msg[c][0];
                    msgSum[1][k] += msg[c][1];
                }
            }
            // find the local maximum given y_j = 0 or 1
            for ( int k = 0; k < powNumJ; k ++ ) {
                if (msgSum[0][k] >= msgSum[1][k]) {
                    msg[j][k] = msgSum[0][k];
                    yMax[j][k] = 0.0;
                } else {
                    msg[j][k] = msgSum[1][k];
                    yMax[j][k] = 1.0;
                }
            }
        }

        // Step 3: find the y maximizing the joint probability  (from root(s) to leaf(s))
        int indexJ;
        y = new double[L];
        for (  int j : m_Chain ) {
            if ( pa[j].length == 0) {
                indexJ = 0;
            } else {
                indexJ = (int)y[pa[j][0]];
            }
            y[j] = yMax[j][indexJ];
        }
        return y;
    }

    private boolean push(double y[], int j) {
        if (j < 0 ) {
            return true;
        }
        else if (y[j] < 1) {
            y[j]++;
            return false;
        }
        else {
            y[j] = 0.0;
            return push(y,--j);
        }
    }
    // ***************************************************************************************
    // ***************************************************************************************
    // ***************************************************************************************
}
