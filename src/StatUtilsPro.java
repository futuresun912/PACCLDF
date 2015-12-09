/**
 * Created by sunlu on 12/9/15.
 */

import meka.core.StatUtils;
import weka.core.Instances;


public class StatUtilsPro extends StatUtils{

    // return the minimum entropy of Y_j and Y_k.
    // min { H(Y_j), H(Y_k) }
    public static double minH(double[][] P, int j, int k) {
        double H_j, H_k;
        double p_j = P[j][j];
        double p_k = P[k][k];
        H_j = - p_j * Math.log( p_j ) -
                ( 1 - p_j ) * Math.log( 1 - p_j );
        H_k = - p_k * Math.log( p_k ) -
                ( 1 - p_k ) * Math.log( 1 - p_k );
        return H_j < H_k ? H_j : H_k;
    }

    // calculate the normalized mutual information between Y_j and Y_k
    public static double NI(double[][] P, int j, int k) {
        double NI = 0.0;
        double p_j = P[j][j];
        double p_k = P[k][k];
        double p_jk = P[j][k];

        // NI(1;1)
        NI += p_jk * Math.log( p_jk / (p_j * p_k) );
        // NI(1;0)
        if ( p_j != p_jk )  // 0 * log(0) = 0
            NI += ( p_j - p_jk ) * Math.log( (p_j-p_jk) / (p_j*(1-p_k)) );
        // NI(0;1)
        if ( p_k != p_jk )  // 0 * log(0) = 0
            NI += ( p_k - p_jk ) * Math.log( (p_k-p_jk) / ((1-p_j)*p_k) );
        // NI(0;0)
        NI += (1-p_j-p_k + p_jk) * Math.log((1 - p_j - p_k + p_jk) / ((1-p_j)*(1-p_k)) );

        // normalization
        return NI / minH(P, j, k);
    }

    // calculate the normalized mutual information matrix
    public static double[][] NI(double[][] P) {
        int L = P.length;
        double[][] M = new double[L][L];

        for (int j = 0; j < L; j ++)
            for (int k = j+1; k < L; k ++)
                M[j][k] = NI(P, j, k);
        return M;
    }

    // calculate the normalized mutual information
    // NI(Y_j; Y_k) = I(Y_j; Y_k) / min {H(Y_j), H(Y_k)}
    public static double[][] NormMargDep(Instances D) {
        int N = D.numInstances();
        int[][] C = getApproxC(D);
        double[][] P = getP(C, N);
        return NI(P);
    }

    // Calcualte the statistics (Imbalance ratio with its mean and variance) of the dataset
    // Return the IR factor for balancing merits in Wrapper based Feature selection
    public static double[] CalcIRFactor(Instances D) {
        int L = D.classIndex();
        int N = D.numInstances();
        int M = D.numAttributes() - L;
        int[][] countM = StatUtils.getApproxC(D);
        int[] countA = new int[L];        // count of positive instances
        double[] IR = new double[L];      // save the imbalance ratio
        double meanIR, varIR, CVIR;       // mean, variance and Coefficient of Variance of IR
        double[] factor = new double[L];  // return the IR factor for Wrapper

        // get the count and ratio of positive instances for each label with the max count
        int maxA = countM[0][0];
        for (int j = 0; j < L; j ++) {
            countA[j] = countM[j][j];
            if (maxA < countA[j])
                maxA = countA[j];
        }
        for (int j = 0; j < L; j ++)
            if (countA[j] == 0)
                countA[j] = maxA;

        // calculate the label-individual Imbalance Ratio array with its mean
        meanIR = 0.0;
        for (int j = 0; j < L; j++) {
            IR[j] = (double) maxA / (double) countA[j];
            meanIR += IR[j];
        }
        meanIR = meanIR / (double)L;

        // calculate the variance of IR array and CVIR
        varIR = 0.0;
        for (int j = 0; j < L; j ++)
            varIR += Math.pow((IR[j]-meanIR), 2) / (L-1);
        CVIR = Math.pow(varIR,0.5)/meanIR;

        // calculate the IR factor
        double r = 0.05;
        for (int j = 0; j < L; j ++) {
            double temp = Math.exp( meanIR / (IR[j]*(CVIR+1)) );
            factor[j] = 2*r*(temp - 1) / (temp + 1) + r;
        }

        return factor;
    }
}
