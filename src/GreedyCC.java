/**
 * Created by sunlu on 12/9/15.
 */

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.SubsetEvaluator;
import weka.attributeSelection.UnsupervisedSubsetEvaluator;
import weka.core.Instances;
import weka.core.Utils;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;


public class GreedyCC extends GreedyStepwise {

    private int[] m_pa;

    protected void setParentSet(int[] pa) throws Exception {
        this.m_pa = pa;
    }

    /**
     * Searches the attribute subset space by forward selection.
     *
     * @param ASEval the attribute evaluator to guide the search
     * @param data the training instances.
     * @return an array (not necessarily ordered) of selected attribute indexes
     * @throws Exception if the search can't be completed
     */
    // This function was modified for CCDR (taking label correlation into account)
    @Override
    public int[] search(ASEvaluation ASEval, Instances data) throws Exception {

        int i;
        double best_merit = -Double.MAX_VALUE;   // evaluation result
        double temp_best, temp_merit;
        int temp_index = 0;
        BitSet temp_group;                        // save the temp group of features
        boolean parallel = (m_poolSize > 1);
        if (parallel) {
            m_pool = Executors.newFixedThreadPool(m_poolSize);
        }

        if (data != null) { // this is a fresh run so reset
            resetOptions();
            m_Instances = data;
        }
        m_ASEval = ASEval;

        m_numAttribs = m_Instances.numAttributes();  // includes current label and its parents

        if (m_best_group == null) {
            m_best_group = new BitSet(m_numAttribs);
        }

        if (!(m_ASEval instanceof SubsetEvaluator)) {
            throw new Exception(m_ASEval.getClass().getName() + " is not a "
                    + "Subset evaluator!");
        }

        int[] paIndices;
        if (m_pa.length > 0) {
            paIndices = Utils.sort(m_pa);
            m_starting = paIndices.clone();
        }

        m_startRange.setUpper(m_numAttribs - 1);   // the range of searching space (remove the parents?)

        if (!(getStartSet().equals(""))) {
            m_starting = m_startRange.getSelection();
        }

        if (m_ASEval instanceof UnsupervisedSubsetEvaluator) {
            m_hasClass = false;
            m_classIndex = -1;
        } else {
            m_hasClass = true;
            m_classIndex = m_Instances.classIndex();   // equals to j not L
        }

        final SubsetEvaluator ASEvaluator = (SubsetEvaluator) m_ASEval;

        if (m_rankedAtts == null) {
            m_rankedAtts = new double[m_numAttribs][2];
            m_rankedSoFar = 0;
        }

        // If a starting subset has been supplied, then initialize the bitset
        if (m_starting != null && m_rankedSoFar <= 0) {   // m_starting : int[]
            for (i = 0; i < m_starting.length; i++) {
                if ((m_starting[i]) != m_classIndex) {
                    m_best_group.set(m_starting[i]);
                }
            }
        } else {
            if (m_backward && m_rankedSoFar <= 0) {
                for (i = 0; i < m_numAttribs; i++) {
                    if (i != m_classIndex) {
                        m_best_group.set(i);
                    }
                }
            }
        }

        best_merit = -1000.0;

        // main search loop
        // search program starts here!
        boolean done = false;
        boolean addone = false;
        boolean z;

        if (m_debug && parallel) {
            System.err.println("Evaluating subsets in parallel...");
        }
        while (!done) {
            List<Future<Double[]>> results = new ArrayList<Future<Double[]>>();
            temp_group = (BitSet) m_best_group.clone();
            temp_best = best_merit;
            if (m_doRank) {
                temp_best = -Double.MAX_VALUE;
            }
            done = true;
            addone = false;


            for (i = m_pa.length; i < m_numAttribs; i++) {
                if (m_backward) {
                    z = ((i != m_classIndex) && (temp_group.get(i)));
                } else {
                    z = ((i != m_classIndex) && (!temp_group.get(i)));
                }
                if (z) {
                    // set/unset the bit
                    if (m_backward) {
                        temp_group.clear(i);
                    } else {
                        temp_group.set(i);
                    }

                    if (parallel) {
                        final BitSet tempCopy = (BitSet) temp_group.clone();
                        final int attBeingEvaluated = i;

                        // make a copy if the evaluator is not thread safe
                        final SubsetEvaluator theEvaluator = (ASEvaluator instanceof weka.core.ThreadSafe) ? ASEvaluator
                                : (SubsetEvaluator) ASEvaluation.makeCopies(m_ASEval, 1)[0];

                        Future<Double[]> future = m_pool.submit(new Callable<Double[]>() {
                            @Override
                            public Double[] call() throws Exception {
                                Double[] r = new Double[2];

                                double e = theEvaluator.evaluateSubset(tempCopy);
                                r[0] = new Double(attBeingEvaluated);
                                r[1] = e;
                                return r;
                            }
                        });

                        results.add(future);
                    } else {
                        temp_merit = ASEvaluator.evaluateSubset(temp_group);
                        if (m_backward) {
                            z = (temp_merit >= temp_best);
                        } else {
                            if (m_conservativeSelection) {
                                z = (temp_merit >= temp_best);
                            } else {
                                z = (temp_merit > temp_best);
                            }
                        }

                        if (z) {
                            temp_best = temp_merit;
                            temp_index = i;
                            addone = true;
                            done = false;
                        }
                    }

                    // unset this addition/deletion
                    if (m_backward) {
                        temp_group.set(i);
                    } else {
                        temp_group.clear(i);
                    }
                    if (m_doRank) {
                        done = false;
                    }
                }
            }

            if (parallel) {
                for (int j = 0; j < results.size(); j++) {
                    Future<Double[]> f = results.get(j);

                    int index = f.get()[0].intValue();
                    temp_merit = f.get()[1].doubleValue();

                    if (m_backward) {
                        z = (temp_merit >= temp_best);
                    } else {
                        if (m_conservativeSelection) {
                            z = (temp_merit >= temp_best);
                        } else {
                            z = (temp_merit > temp_best);
                        }
                    }

                    if (z) {
                        temp_best = temp_merit;
                        temp_index = index;
                        addone = true;
                        done = false;
                    }
                }
            }

            if (addone) {
                if (m_backward) {
                    m_best_group.clear(temp_index);
                } else {
                    m_best_group.set(temp_index);
                }
                best_merit = temp_best;
                if (m_debug) {
                    System.err.print("Best subset found so far: ");
                    int[] atts = attributeList(m_best_group);
                    for (int a : atts) {
                        System.err.print("" + (a + 1) + " ");
                    }
                    System.err.println("\nMerit: " + best_merit);
                }
                m_rankedAtts[m_rankedSoFar][0] = temp_index;
                m_rankedAtts[m_rankedSoFar][1] = best_merit;
                m_rankedSoFar++;
            }
        }

        if (parallel) {
            m_pool.shutdown();
        }

        m_bestMerit = best_merit;
        return attributeList(m_best_group);
    }
}

