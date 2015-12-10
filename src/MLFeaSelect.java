/**
 * Created by sunlu on 12/9/15.
 */

import meka.core.A;
import meka.core.F;
import meka.core.MLUtils;
import weka.attributeSelection.*;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;


public class MLFeaSelect {

    protected int L;
    protected boolean m_FlagRanker;
    protected boolean m_IG;
    protected boolean m_CFS;
    protected double m_PercentFeature;
    protected int[][] m_Indices1;
    protected int[] m_Indices2;
    protected int[][] m_Indices;
    protected Instances[] m_dataHeader;
    protected Instance[] m_instTemplate;

    public MLFeaSelect(int L) {
        this.L = L;
        this.m_FlagRanker = false;
        this.m_IG = false;
        this.m_CFS = false;
        this.m_PercentFeature = 0.2;
        this.m_Indices1 = new int[L][];
        this.m_Indices2 = new int[]{};
        this.m_Indices = new int[L][];
        this.m_dataHeader = new Instances[L];
        this.m_instTemplate = new Instance[L];
    }

    protected void setPercentFeature(double fraction) throws Exception {
        this.m_PercentFeature = fraction;
    }

    protected int getNumFeaCfs(int j) throws Exception {
        return m_Indices2.length;
    }

    // The first-stage feature selection for MLC
    protected void feaSelect1(Instances D, int num) throws Exception {

        int d_cut = (int) ((D.numAttributes() - L) * m_PercentFeature);

        // Perform FS for each label
        for (int j = 0; j < num; j++) {

            int[] pa = new int[0];
            pa = A.append(pa, j);
            Instances D_j = MLUtils.keepAttributesAt(new Instances(D), pa, L);
            D_j.setClassIndex(0);

            AttributeSelection selector = new AttributeSelection();
            InfoGainAttributeEval evaluator = new InfoGainAttributeEval();
            Ranker searcher = new Ranker();
            searcher.setNumToSelect(d_cut);
            selector.setEvaluator(evaluator);
            selector.setSearch(searcher);

            // Obtain the indices of selected features
            selector.SelectAttributes(D_j);
            m_Indices1[j] = selector.selectedAttributes();
            // Sort the selected features for the Ranker
            m_FlagRanker = true;
            m_Indices1[j] = shiftIndices(m_Indices1[j], L, pa);
        }
        m_IG = true;
    }

    // The second-stage feature selection for MLC
    protected void feaSelect2(Instances D_j, int j) throws Exception {
        // Remove all the labels except j and its parents
        int[] pa = new int[0];
        D_j.setClassIndex(j);
        pa = A.append(pa, j);
        Instances tempD = MLUtils.keepAttributesAt(new Instances(D_j), pa, L);

        // Initialization of the feature selector
        AttributeSelection selector = new AttributeSelection();
        CfsSubsetEval evaluator = new CfsSubsetEval();
        GreedyStepwise searcher = new GreedyStepwise();
        searcher.setSearchBackwards(true);
        selector.setEvaluator(evaluator);
        selector.setSearch(searcher);

        // Obtain the indices of selected features
        selector.SelectAttributes(tempD);
        m_Indices2 = selector.selectedAttributes();
        m_Indices2 = shiftIndices(m_Indices2, L, pa);
        m_CFS = true;
    }

    // MLIG
    protected void feaSelect1IR(Instances D, int num, double[] factor) throws Exception {

        int d_cut = (int) ((D.numAttributes() - L) * m_PercentFeature);

        // Perform FS for each label
        for (int j = 0; j < num; j++) {

            int[] pa = new int[0];
            pa = A.append(pa, j);
            Instances D_j = MLUtils.keepAttributesAt(new Instances(D), pa, L);
            D_j.setClassIndex(0);

            AttributeSelection selector = new AttributeSelection();
            InfoGainAttributeEval evaluator = new InfoGainAttributeEval();
            Ranker searcher = new Ranker();
            searcher.setNumToSelect((int)(d_cut*(1.0+factor[j])));
            selector.setEvaluator(evaluator);
            selector.setSearch(searcher);

            // Obtain the indices of selected features
            selector.SelectAttributes(D_j);
            m_Indices1[j] = selector.selectedAttributes();
            // Sort the selected features for the Ranker
            m_FlagRanker = true;
            m_Indices1[j] = shiftIndices(m_Indices1[j], L, pa);
        }
        m_IG = true;
    }

    // MLCFS
    protected void feaSelect2PA(Instances D_j, int j) throws Exception {
        // Remove all the labels except j and its parents
        int[] pa = new int[0];
        D_j.setClassIndex(j);
        pa = A.append(pa, j);
        Instances tempD = MLUtils.keepAttributesAt(new Instances(D_j), pa, L);

        // Initialization of the feature selector
        AttributeSelection selector = new AttributeSelection();
        CfsSubsetEval evaluator = new CfsSubsetEval();
        GreedyCC searcher = new GreedyCC();
        searcher.setParentSet(pa);
        searcher.setConservativeForwardSelection(true);
        selector.setEvaluator(evaluator);
        selector.setSearch(searcher);

        // Obtain the indices of selected features
        selector.SelectAttributes(tempD);
        m_Indices2 = selector.selectedAttributes();
        m_Indices2 = shiftIndices(m_Indices2, L, pa);
        m_CFS = true;
    }


    // Shift the indices of selected features for filtering D
    protected int[] shiftIndices(int[] indices, int L, int[] pa) throws Exception {

        // Remove the current label j
        indices = A.delete(indices, indices.length-1);

        // remove the parent labels from m_Indices2[j] for post-processing
        if (pa.length > 1) {
            int[] indexTemp = new int[0];
            for (int j = 0; j < indices.length; j ++)
                if (indices[j] >= pa.length)
                    indexTemp = A.append(indexTemp, indices[j]);
            indices = indexTemp.clone();   // possible problem
        }

        // Shift indices of features by taking labels into account
        for (int j = 0; j < indices.length; j ++)
            indices[j] = indices[j] + L - pa.length;

        // Sort the feature indices in ascending order
        if (m_FlagRanker) {
            int[] temp1 = Utils.sort(indices);
            int[] temp2 = new int[indices.length];
            for (int j = 0; j < indices.length; j ++) {
                temp2[j] = indices[temp1[j]];
            }
            indices = temp2.clone();
        }
        return indices;
    }

    // Transform an instances based on the indices of selected features
    protected Instances instTransform(Instances D, int j) throws Exception {
        int L = D.classIndex();
        if (m_IG && m_CFS) {
            // m_Indices <-- m_Indices1( m_Indices2 )
            if ( m_Indices2.length == 0 ) {
                m_Indices[j] = m_Indices1[j].clone();
            } else {
                int[] indexTemp = new int[m_Indices2.length];
                for (int k = 0; k < m_Indices2.length; k++)
                    indexTemp[k] = m_Indices1[j][m_Indices2[k]-L];
                m_Indices[j] = indexTemp.clone();
            }
            m_CFS = false;
        } else if (m_IG) {
            m_Indices[j] = m_Indices1[j].clone();
        } else if (m_CFS) {
            m_Indices[j] = m_Indices2.clone();
        }

        int[] index_j = new int[m_Indices[j].length+L];
        for (int k = 0; k < index_j.length; k ++) {
            if (k < L) {
                index_j[k] = k;
            }
            else {
                index_j[k] = m_Indices[j][k - L];
            }
        }

        D.setClassIndex(-1);
        Instances D_j = F.remove(D, index_j, true);
        D_j.setClassIndex(L);
        D.setClassIndex(L);

        m_instTemplate[j] = (Instance)D_j.instance(0).copy();
        m_instTemplate[j].setDataset(null);
        for (int k = 0; k < L; k ++) {
            m_instTemplate[j].deleteAttributeAt(0);
        }
        m_dataHeader[j] = new Instances(D_j, 0);
        return D_j;
    }

    // Transform an test instance based on the indices of selected features
    protected Instance instTransform(Instance x_j, int j) throws Exception {

        int L = x_j.classIndex();
        Instance x_out = (Instance)m_instTemplate[j].copy();
        x_out.setDataset(null);
        for (int k = 0; k < L; k ++) {
            x_out.insertAttributeAt(0);
        }
        copyInstValues(x_out, x_j, m_Indices[j], L);
        x_out.setDataset(m_dataHeader[j]);
        return x_out;
    }


    /**
     * CopyValues - Set x_dest[i++] = x_src[j] for all j in indices[].
     */
    public Instance copyInstValues(Instance x_dest, Instance x_src,
                                   int indices[], int L) {
        int i = L;
        for(int j : indices) {
            x_dest.setValue(i++,x_src.value(j));
        }
        return x_dest;
    }

}

