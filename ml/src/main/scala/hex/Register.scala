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
package hex

import org.apache.spark.ml.spark.models.svm.SVM
import water.H2O
import water.api.{GridSearchHandler, ModelBuilderHandler}

class Register extends water.api.AbstractRegister {

  @throws[ClassNotFoundException]
  override def register(relativeResourcePath: String) = {

    val models = Seq(new SVM(true))

    // TODO delicious copy-pasta from h2o-3 hex.api.Register, maybe possible to refactor that part?
    for (algo <- models) {
      val base: String = algo.getClass.getSimpleName
      val lbase: String = base.toLowerCase
      val bh_clz: Class[_] = classOf[ModelBuilderHandler[_, _, _]]
      // TODO shouldn't this be hardcoded somewhere in one place in h2o-3?? can't find
      val version: Int = 3
      H2O.registerPOST("/" + version + "/ModelBuilders/" + lbase, bh_clz, "train", "Train a " + base + " model.")
      H2O.registerPOST(
        "/" + version + "/ModelBuilders/" + lbase + "/parameters",
        bh_clz,
        "validate_parameters",
        "Validate a set of " + base + " model builder parameters."
      )
      // Grid search is experimental feature
      H2O.registerPOST(
        "/99/Grid/" + lbase,
        classOf[GridSearchHandler[_,_,_,_]],
        "train",
        "Run grid search for " + base + " model."
      )
    }
  }

}
