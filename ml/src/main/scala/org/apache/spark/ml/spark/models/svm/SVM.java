/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.spark.ml.spark.models.svm;

import hex.*;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.SparkContext;
import org.apache.spark.h2o.H2OContext;
import org.apache.spark.ml.spark.models.svm.SVMModel.SVMOutput;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import water.Scope;
import water.fvec.Frame;
import water.fvec.H2OFrame;
import water.fvec.Vec;
import water.util.Log;

public class SVM extends ModelBuilder<SVMModel, SVMParameters, SVMOutput> {

    public SVM(boolean startup_once) {
        super(new SVMParameters(), startup_once);
    }

    public SVM(SVMParameters parms) {
        super(parms);
        init(false);
    }

    @Override
    protected Driver trainModelImpl() {
        return new SVMDriver();
    }

    @Override
    public ModelCategory[] can_build() {
        return new ModelCategory[]{
                ModelCategory.Binomial,
                ModelCategory.Regression
        };
    }

    @Override
    public boolean isSupervised() {
        return true;
    }

    @Override
    public void init(boolean expensive) {
        super.init(expensive);

        _parms.validate(this);

        if (_train == null) return;

        if (null != _parms._initial_weights) {
            Frame user_points = _parms._initial_weights.get();
            if (user_points.numCols() != _train.numCols() - numSpecialCols()) {
                error("_initial_weights",
                        "The user-specified initial weights must have the same number of columns " +
                                "(" + (_train.numCols() - numSpecialCols()) + ") as the training observations");
            }

            if(user_points.hasNAs()) {
                error("_initial_weights", "Initial weights cannot contain missing values.");
            }
        }

        if(_train.hasNAs()) {
            error("_train", "Training frame cannot contain any missing values.");
        }

        for(Vec vec : _train.vecs()) {
            if( !(vec.isNumeric() || vec.isTime() || vec.isCategorical()) ) {
                error("_train", "SVM supports only frames with numeric values. But a " + vec.get_type_str() + " was found.");
            }
        }

        if(null != _parms._response_column && null == _train.vec(_parms._response_column)) {
            error("_train", "Training frame has to contain the response column.");
        }


        if( _train != null && _parms._response_column != null) {
            String[] responseDomains = responseDomains();
            if (null == responseDomains) {
                if (!(Double.isNaN(_parms._threshold))) {
                    error("_threshold", "Threshold cannot be set for regression SVM. Set the threshold to NaN or modify the response column to an enum.");
                }

                if (!_train.vec(_parms._response_column).isNumeric()) {
                    error("_threshold", "Threshold cannot be set for regression SVM. Set the threshold to NaN or modify the response column to an enum.");
                    error("_response_column", "Regression SVM requires the response column type to be numeric.");
                }
            } else {
                if (Double.isNaN(_parms._threshold)) {
                    error("_threshold", "Threshold has to be set for binomial SVM. Set the threshold to a numeric value or change the response column type.");
                }

                if (responseDomains.length != 2 ||
                        (
                                (!responseDomains[0].equals("0") && !responseDomains[1].equals("1")) &&
                                        (!responseDomains[0].equals("1") && !responseDomains[1].equals("0")))
                        ) {
                    error("_response_column", "Binomial SVM requires the response column's domain to be \"0\" and \"1\".");
                }
            }
        }
    }

    private String[] responseDomains() {
        int idx = _parms.train().find(_parms._response_column);
        if(idx == -1) {
            return null;
        }
        return _parms.train().domains()[idx];
    }

    @Override
    public int numSpecialCols() {
        return (hasOffsetCol() ? 1 : 0) +
                (hasWeightCol() ? 1 : 0) +
                (hasFoldCol() ? 1 : 0) + 1;
    }

    private final class SVMDriver extends Driver {

        transient private SparkContext sc = H2OContext.getSparkContext();
        transient private H2OContext h2oContext = H2OContext.getOrCreate(sc);
        transient private SQLContext sqlContext = SQLContext.getOrCreate(sc);

        @Override
        public void compute2() {
            SVMModel model = null;
            try {
                Scope.enter();
                _parms.read_lock_frames(_job);
                init(true);

                // The model to be built
                model = new SVMModel(dest(), _parms, new SVMModel.SVMOutput(SVM.this));
                model.delete_and_lock(_job);

                RDD<LabeledPoint> training = getTrainingData(
                        _train,
                        _parms._response_column,
                        model._output.nfeatures()
                );
                training.cache();

                SVMWithSGD svm = new SVMWithSGD();
                svm.setIntercept(_parms._add_intercept);

                svm.optimizer().setNumIterations(_parms._max_iterations);

                svm.optimizer().setStepSize(_parms._step_size);
                svm.optimizer().setRegParam(_parms._reg_param);
                svm.optimizer().setMiniBatchFraction(_parms._mini_batch_fraction);
                svm.optimizer().setConvergenceTol(_parms._convergence_tol);
                svm.optimizer().setGradient(_parms._gradient.get());
                svm.optimizer().setUpdater(_parms._updater.get());

                final org.apache.spark.mllib.classification.SVMModel trainedModel =
                        (null == _parms._initial_weights) ?
                                svm.run(training) :
                                svm.run(training, vec2vec(_parms.initialWeights().vecs()));
                training.unpersist(false);

                model._output.weights_$eq(trainedModel.weights().toArray());
                model._output.iterations_$eq(_parms._max_iterations);
                model._output.interceptor_$eq(trainedModel.intercept());

                model._output._training_metrics =
                        SVMDriverUtil.computeMetrics(model, training, trainedModel, _parms.train(), responseDomains());
                model.update(_job);

                _job.update(model._parms._max_iterations);

                if (_valid != null) {
                    model.score(_parms.valid()).delete();
                    model._output._validation_metrics = ModelMetrics.getFromDKV(model,_parms.valid());
                    model.update(_job);
                }

                model._output.interceptor_$eq(trainedModel.intercept());

                Log.info(model._output._model_summary);
            } finally {
                if (model != null) model.unlock(_job);
                _parms.read_unlock_frames(_job);
                Scope.exit();
            }
            tryComplete();
        }

        private Vector vec2vec(Vec[] vals) {
            double[] dense = new double[vals.length];
            for (int i = 0; i < vals.length; i++) {
                dense[i] = vals[i].at(0);
            }
            return Vectors.dense(dense);
        }

        private RDD<LabeledPoint> getTrainingData(Frame parms, String _response_column, int nfeatures) {
            return h2oContext.createH2OSchemaRDD(new H2OFrame(parms), sqlContext)
                    .javaRDD()
                    .map(new RowToLabeledPoint(nfeatures, _response_column)).rdd();
        }
    }
}

class RowToLabeledPoint implements Function<Row, LabeledPoint> {
    private final int nfeatures;
    private final String _response_column;

    RowToLabeledPoint(int nfeatures, String response_column) {
        this.nfeatures = nfeatures;
        this._response_column = response_column;
    }

    @Override
    public LabeledPoint call(Row row) throws Exception {
        StructField[] fields = row.schema().fields();
        double[] features = new double[nfeatures];
        for (int i = 0; i < nfeatures; i++) {
            features[i] = toDouble(row.get(i), fields[i]);
        }

        int responseColIndx = row.schema().fieldIndex(_response_column);

        return new LabeledPoint(
                toDouble(row.getAs(_response_column), fields[responseColIndx]),
                Vectors.dense(features));
    }

    private double toDouble(Object value, StructField fieldStruct) {
        if(fieldStruct.dataType().sameType(DataTypes.ByteType)) {
            return ((Byte)value).doubleValue();
        }

        if(fieldStruct.dataType().sameType(DataTypes.ShortType)) {
            return ((Short)value).doubleValue();
        }

        if(fieldStruct.dataType().sameType(DataTypes.IntegerType)) {
            return ((Integer)value).doubleValue();
        }

        if(fieldStruct.dataType().sameType(DataTypes.DoubleType)) {
            return (Double)value;
        }

        if(fieldStruct.dataType().sameType(DataTypes.StringType)) {
            return Double.parseDouble(value.toString());
        }

        throw new IllegalArgumentException("Target column has to be an enum or a number. " + fieldStruct.toString());
    }
}

