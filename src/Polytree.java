/**
 * Created by sunlu on 12/9/15.
 */

import meka.classifiers.multilabel.cc.CNode;
import meka.core.A;
import meka.core.M;
import mst.Edge;
import mst.EdgeWeightedGraph;
import mst.KruskalMST;
import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import java.util.Arrays;


public class Polytree {

    private double para1;    // for miThreshold
    private int para2;       // set as max|pa(.)|
    private double[][] CD;
    private boolean[] flagCB;
    private int numVisited;
    private boolean[] visited;
    private int L;
    protected int[] chainOrder;
    protected int numFolds;
    protected boolean depMode;

    public Polytree () {
        this.para1 = 0.2;
        this.para2 = 6;
        this.numFolds = 3;
        this.depMode = false;
    }

    protected void setPara1(double fraction) throws Exception {
        this.para1 = fraction;
    }

    protected void setPara2(int maxPa) throws Exception {
        this.para2 = maxPa;
    }

    protected void setNumFolds(int n) throws Exception {
        this.numFolds = n;
    }

    protected int[] getChainOrder() throws Exception {
        return chainOrder;
    }

    protected void setDepMode(boolean mode) throws Exception {
        this.depMode = mode;
    }

    // Build a polytree on the tree
    protected int[][] polyTree(Instances D, Instances[] newD) throws Exception {
        L = (D==null) ? newD[0].classIndex() : D.classIndex();
        CD = new double[L][L];
        numVisited = 0;
        int root = 0;
        int[][] pa = new int[L][0];
        visited = new boolean[L];
        flagCB = new boolean[L];
        Arrays.fill(visited, false);
        Arrays.fill(flagCB, false);

        if (depMode) {
            // Calculate the conditional MI matrix
            if (newD == null)
                CD = conDepMatrix(D);
            if (newD != null)
                CD = conDepMatrix(newD);
        } else {
            // Calculate the marginal normalized MI matrix
            CD = StatUtilsPro.NormMargDep(D);
        }

        // Build the tree skeleton
        int[][] paTree = skeleton(CD);

        // Find the causal basins
        int[][] paPoly = new int[L][L];
        causalBasin(root, paTree, paPoly);

        // If causal basin can't cover all labels, build a directed tree (paTemp)
        int[][] paTemp = new int[L][0];
        root = -1;
        for (int j = 0; j < L; j ++) {
            for (int k = j; k < L; k ++){
                if (paPoly[j][k] == 1) {
                    root = j;
                    Arrays.fill(visited, false);
                    visited[root] = true;
                    treeify(root, paPoly, paTemp);
                    break;
                }
            }
            if (root != -1) break;
        }
        // Save the parents of every node in the polytree (pa)
        for (int j = 0; j < L; j ++) {
            for (int k = j; k < L; k ++) {
                if (paPoly[j][k] == 3)
                    pa[j] = A.append(pa[j], k);
                if (paPoly[j][k] == 2)
                    pa[k] = A.append(pa[k], j);
            }
        }
        for (int j = 0; j < L; j ++) {
            if (pa[j].length < 1 ) {
                for (int v : paTemp[j]) {
                    pa[j] = A.append(pa[j], v);
                    paPoly[j][v] = 3;
                    paPoly[v][j] = 2;
                }
            }
        }

        // Rank the labels in the polytree (rank)
        root = 0;
        int[] rank = new int[L];
        Arrays.fill(rank, 0);
        Arrays.fill(visited, false);
        rankLabel(root, paPoly, rank);
        chainOrder = Utils.sort(rank);

        // Enhance the polytree
        int[] temp = new int[]{};
        double thCD = 0.0005;
        for (int j : chainOrder) {
            for (int k : temp) {
                if (paPoly[j][k] != 3) {
                    if (j < k && CD[j][k] > thCD)
                        pa[j] = A.append(pa[j], k);
                    if (j > k && CD[k][j] > thCD)
                        pa[j] = A.append(pa[j], k);
                }
            }
            temp = A.append(temp, j);
        }
        return pa;
    }


    // Calculation of conditional dependence between labels given the observation of instances
    protected double[][] conDepMatrix(Instances newData) throws Exception {

        int L = newData.classIndex();
        CNode[][] miNodes = new CNode[L][];
        int[] paNode;
        Classifier model = new Logistic();
        double MI[][] = new double[L][L];

        // n-fold CV for calculation of MI matrix
        for(int i = 0; i < numFolds; i++) {

            Instances[] D_train = new Instances[L];
            Instances[] D_test = new Instances[L];
            for (int j = 0; j < L; j ++) {
                D_train[j] = newData.trainCV(numFolds, i);
                D_test[j] = newData.testCV(numFolds, i);
            }

            // train L*(L+1)/2 logistic classifiers for calculating conditional probability.
            for (int j = 0; j < L; j ++) {
                miNodes[j] = new CNode[L-j];
                paNode = new int[]{};
                for (int k = j; k < L; k ++) {
                    if (k != j)
                        paNode = A.append(paNode, k);
                    miNodes[j][k-j] = new CNode(j, null, paNode);
                    miNodes[j][k-j].build(D_train[j], model);
                    if (k != j)
                        paNode = A.delete(paNode, 0);
                }
            }

            // calculate the conditional mutual information
            for (int j = 0; j < L; j ++)
                for (int k = j + 1; k < L; k ++ )
                    MI[j][k] = conMI(D_test[j], D_test[k], miNodes, j, k);
            MI = addMatrix(MI, MI);
        }
        return MI;
    }

    // Calculation of conditional dependence between labels given the observation of instances
    protected double[][] conDepMatrix(Instances[] newData) throws Exception {

        int L = newData[0].classIndex();
        CNode[][] miNodes = new CNode[L][];
        int[] paNode;
        Classifier model = new Logistic();
        double MI[][] = new double[L][L];

        // n-fold CV for calculation of MI matrix
        for(int i = 0; i < numFolds; i++) {

            Instances[] D_train = new Instances[L];
            Instances[] D_test = new Instances[L];
            for (int j = 0; j < L; j ++) {
                D_train[j] = newData[j].trainCV(numFolds, i);
                D_test[j] = newData[j].testCV(numFolds, i);
            }

            // train L*(L+1)/2 logistic classifiers for calculating conditional probability.
            for (int j = 0; j < L; j ++) {
                miNodes[j] = new CNode[L-j];
                paNode = new int[]{};
                for (int k = j; k < L; k ++) {
                    if (k != j)
                        paNode = A.append(paNode, k);
                    miNodes[j][k-j] = new CNode(j, null, paNode);
                    miNodes[j][k-j].build(D_train[j], model);
                    if (k != j)
                        paNode = A.delete(paNode, 0);
                }
            }

            // calculate the conditional mutual information
            for (int j = 0; j < L; j ++)
                for (int k = j + 1; k < L; k ++ )
                    MI[j][k] = conMI(D_test[j], D_test[k], miNodes, j, k);
            MI = addMatrix(MI, MI);
        }
        return MI;
    }


    // use the learned classifiers to get conditional probability
    protected double conMI(Instances D_j, Instances D_k, CNode[][] miNodes, int j, int k) throws Exception {

        int L = D_j.classIndex();
        int N = D_j.numInstances();
        double y[] = new double[L];
        double I = 0.0;       		 	 // conditional mutual information for y_j and y_k
        double p_1, p_2;      			 // p( y_j = 1 | x ), p( y_j = 2 | x )
        double p_12[] = {0.0,0.0};       // p_12[0] = p( y_j = 1 | y_k = 0, x ) and p_12[1] = p( y_j = 1 | y_k = 1, x )

        for (int i = 0; i < N; i ++) {
            Arrays.fill(y, 0);
            p_1 = Math.max( miNodes[j][0].distribution((Instance)D_j.instance(i).copy(), y)[1], 0.000001 );           // p( y_j = 1 | x )
            p_1 = Math.min(p_1, 0.999999);
            p_1 = Math.max(p_1, 0.000001);
            Arrays.fill(y, 0);
            p_2 = Math.max( miNodes[k][0].distribution((Instance)D_k.instance(i).copy(), y)[1], 0.000001 );           // p( y_k = 1 | x )
            p_2 = Math.min(p_2, 0.999999);
            p_2 = Math.max(p_2, 0.000001);
            Arrays.fill(y, 0);
            p_12[0] = Math.max( miNodes[j][k-j].distribution((Instance)D_j.instance(i).copy(), y)[1], 0.000001 );     // p( y_j = 1 | y_k = 0, x )
            p_12[0] = Math.min(p_12[0], 0.999999);
            p_12[0] = Math.max(p_12[0], 0.000001);
            Arrays.fill(y, 0);
            Arrays.fill(y, k, k+1, 1.0);
            p_12[1] = Math.max( miNodes[j][k-j].distribution((Instance)D_j.instance(i).copy(), y)[1], 0.000001 );     // p( y_j = 1 | y_k = 1, x )
            p_12[1] = Math.min(p_12[1], 0.999999);
            p_12[1] = Math.max(p_12[1], 0.000001);

            I += ( 1 - p_12[0] ) * ( 1 - p_2 ) * Math.log( ( 1 - p_12[0] ) / ( 1 - p_1 ) );     // I( y_j = 0 ; y_k = 0 )
            I += ( 1 - p_12[1] ) * (     p_2 ) * Math.log( ( 1 - p_12[1] ) / ( 1 - p_1 ) );     // I( y_j = 0 ; y_k = 1 )
            I += (     p_12[0] ) * ( 1 - p_2 ) * Math.log( (     p_12[0] ) / (     p_1 ) );     // I( y_j = 1 ; y_k = 0 )
            I += (     p_12[1] ) * (     p_2 ) * Math.log( (     p_12[1] ) / (     p_1 ) );     // I( y_j = 1 ; y_k = 0 )
        }
        I = I / N;
        return I;
    }

    // C = A + B
    protected double[][] addMatrix(double[][] A, double[][] B) {

        double[][] C = new double[A.length][A[0].length];

        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < A[i].length; j++ )
                C[i][j] = A[i][j] + B[i][j];

        return C;
    }


    // Learning of the tree skeleton (paTree)
    protected int[][] skeleton(double[][] CD) throws Exception {

        CD = M.multiply(CD, -1);
        EdgeWeightedGraph G = new EdgeWeightedGraph(L);
        for(int i = 0; i < L; i ++) {
            for(int j = i+1; j < L; j++) {
                Edge e = new Edge(i, j, CD[i][j]);
                G.addEdge(e);
            }
        }

        KruskalMST mst = new KruskalMST(G);
        int[][] paTree =new int[L][0];
        for (int j = 0; j < L; j ++)
            paTree[j] = new int[]{};
        for(Edge e : mst.edges()) {
            int j = e.either();
            int k = e.other(j);
            paTree[j] = A.append(paTree[j], k);
            paTree[k] = A.append(paTree[k], j);
        }
        return paTree;
    }


    // Find possible causal basins based on the tree skeleton
    // paPoly contains three types of dependence: connected(1),
    // children(2) and parents(3). (check it from rows not columns)
    private void causalBasin(int root, int[][] paTree, int[][] paPoly) throws Exception {
        if ( !visited[root] ) {
            if ( paTree[root].length == 1 ) {      // if root isn't a multi-edge node
                if (paPoly[root][paTree[root][0]] == 0) {
                    paPoly[root][paTree[root][0]] = 1;
                }
                visited[root] = true;
                numVisited ++;
            } else {                              // if root is a multi-edge (more than two) node
                zeroMI(root, paTree,paPoly);
            }
        }

        for (int connectedNode : paTree[root]) {  // Recursively perform on connected nodes
            if (!visited[connectedNode])
                causalBasin(connectedNode, paTree, paPoly);
            else if (numVisited == visited.length)
                break;
            else
                continue;
        }
    }


    private void zeroMI(int root, int[][] paTree, int[][] paPoly) throws Exception {
        // Calculation the threshold
        double min = 1.0;
        for (int j : paTree[root]) {
            if (root < j)
                min = CD[root][j] < min ? CD[root][j] : min;
            else
                min = CD[j][root] < min ? CD[j][root] : min;
        }
        double miThreshold = para1 * min;

        // Find all the candidates parents by thresholding
        int[] paTemp = new int[]{};
        boolean[] selected = new boolean[paPoly[0].length];
        Arrays.fill(selected, false);
        for (int j : paTree[root]) {
            for (int k : paTree[root]) {
                if (j < k) {
                    if( paPoly[root][j] != 2 && paPoly[root][k] != 2 ) {
                        if (CD[j][k] < miThreshold) {
                            if (!selected[j]) {
                                paTemp = A.append(paTemp, j);
                                selected[j] = true;
                            }
                            if (!selected[k]) {
                                paTemp = A.append(paTemp, k);
                                selected[k] = true;
                            }
                        }
                    }
                }
            }
        }

        // 1 if root is a multi-parent node, save its direct parents and children if any
        if (paTemp.length > 1) {
            int maxN = para2;  // set the maximum number of parents
            double[] DepScore = new double[paTemp.length];
            for (int j = 0; j < paTemp.length; j++) {
                for (int k : paTemp) {
                    if (paTemp[j] < k)
                        DepScore[j] = DepScore[j] + CD[paTemp[j]][k];
                    else
                        DepScore[j] = DepScore[j] + CD[k][paTemp[j]];
                }
            }
            double[] copyDepScore = DepScore.clone();
            Arrays.sort(copyDepScore);
            // 1.1 |pa(root)| exceeds the maxN, remove some parents from pa(root)
            if (paTemp.length > maxN) {
                for (int j = 0; j < maxN; j++) {
                    for (int k = 0; k < DepScore.length; k++) {
                        if (copyDepScore[j] == DepScore[k]) {
                            paPoly[root][paTemp[k]] = 3;
                            paPoly[paTemp[k]][root] = 2;
                            flagCB[root] = true;
                        }
                    }
                }
            } else {
                // 1.2 otherwise (pa(root) is small enough), save the pa(root)
                for (int j : paTemp) {
                    paPoly[root][j] = 3;
                    paPoly[j][root] = 2;
                    flagCB[root] = true;
                }
            }
            // save the non-parent nodes as child nodes
            for (int j : paTree[root]) {
                if (paPoly[root][j] != 3) {
                    paPoly[root][j] = 2;
                    paPoly[j][root] = 3;
                    flagCB[j] = true;
                }
            }
            visited[root] = true;
            numVisited ++;
        } else if (flagCB[root]) {   // 2 if CB exists for root, build it following causal flows
            for (int j : paTree[root]) {
                if (paPoly[root][j] != 3) {
                    paPoly[root][j] = 2;
                    paPoly[j][root] = 3;
                    flagCB[j] = true;
                }
            }
            visited[root] = true;
            numVisited ++;
        } else {              // 3 if CB doesn't exist for root, assign undirected edges
            for (int j : paTree[root]) {
                if (paPoly[root][j] == 0)
                    paPoly[root][j] = 1;
            }

            visited[root] = true;
            numVisited ++;
        }
    }

    // Build a directed tree from a root
    private void treeify(int root, int[][] paPoly, int[][] paTemp) throws Exception {
        int children[] = new int[]{};
        for(int j = 0; j < paPoly[root].length; j ++) {
            if (paPoly[root][j] != 0 && !visited[j]) {
                visited[j] = true;
                children = A.append(children, j);
                if (paPoly[root][j] != 3)
                    paTemp[j] = A.append(paTemp[j], root);
            }
        }
        // go through again
        for(int child : children)
            treeify(child, paPoly, paTemp);
    }


    // Rank the labels for the chain order
    private void rankLabel(int root, int[][] paPoly, int[] rank) throws Exception {
        if (!visited[root]) {
            int[] parents = new int[]{};
            int[] children = new int[]{};
            for (int j = 0; j < rank.length; j++) {
                if (paPoly[root][j] == 3) {
                    rank[j] = rank[root] - 1;
                    parents = A.append(parents, j);
                }
                if (paPoly[root][j] == 2) {
                    rank[j] = rank[root] + 1;
                    children = A.append(children, j);
                }
            }
            visited[root] = true;
            for (int parent : parents)
                rankLabel(parent, paPoly, rank);
            for (int child : children)
                rankLabel(child, paPoly, rank);
        }
    }
}
