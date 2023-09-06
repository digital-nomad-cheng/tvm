/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/contrib/arm_compute_lib/acl_runtime.cc
 * \brief A simple JSON runtime for Arm Compute Library.
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

#include "ncnn/net.h"
#include "stdio.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime::json;

class NCNNRuntime : public JSONRuntimeBase {
public: 
  /*!
   * \brief The NCNN runtime module. Deserialize the provided functions 
   * on creation and store in the layer cache.
   * 
   * \param symbol_name The name of the function.
   * \param graph_json serialized JSON representation of a sub-graph.
   * \param const_names The names of each constant in the sub-graph.
   */ 
  explicit NCNNRuntime(const std::string& symbol_name, const std::string& graph_json,
      const Array<String>& const_names) : JSONRuntimeBase(symbol_name, graph_json, const_names) {}
  
  /*!
   * \brief The type key of the mdoule.
   * 
   * \ return module type key
   */ 
  const char* type_key() const override { return "ncnn"; }

  /*!
   * \brief Initialize runtime. Create ncnn layer from JSON representation.
   *
   * \param consts The constant params from compiled model.
   */ 
  void Init(const Array<NDArray>& consts) override {
    ICHECK_EQ(consts.size(), const_idx_.size()) 
      << "The number of input constants must match the number of required.";
    LOG(INFO) << "Initilize ncnn runtime engine"; 
  }
  
  void Run() override {
    LOG(INFO) << "Run ncnn runtime engine";
    ncnn::Mat input(6);

    // fill random
    for (int i = 0; i < input.total(); i++)
    {
        input[i] = rand() % 10;
    }
    
    ncnn::Mat out1;
    inner_product_lowlevel(input, out1);
    printf("Use low level API...\n");
    pretty_print(out1);
  }

  void inner_product_lowlevel(const ncnn::Mat& rgb, ncnn::Mat& out, bool use_bias=false)
  {
      ncnn::Option opt;
      opt.num_threads = 2;

      ncnn::Layer* op = ncnn::create_layer("InnerProduct");

      // set param
      ncnn::ParamDict pd;
      pd.set(0, 3);// num_output
      if (use_bias)
      {
        pd.set(1, 1);// use bias_term
      }
      else 
      {
        pd.set(1, 0);// no bias_term
      }
      pd.set(2, 3);// weight_data_size

      op->load_param(pd);

      // set weights
      ncnn::Mat weights[2];
      weights[0].create(3);// weight_data
      weights[1].create(3);// bias data

      for (int i=0; i<3; i++)
      {
          weights[0][i] = 1.f / 9;
          weights[1][i] = 1.f;
      }


      op->load_model(ncnn::ModelBinFromMatArray(weights));

      op->create_pipeline(opt);

      // forward
      op->forward(rgb, out, opt);

      op->destroy_pipeline(opt);

      delete op;
  }
  void pretty_print(const ncnn::Mat& m)
  {
      for (int q=0; q<m.c; q++)
      {
          const float* ptr = m.channel(q);
          for (int y=0; y<m.h; y++)
          {
              for (int x=0; x<m.w; x++)
              {
                  printf("%f ", ptr[x]);
              }
              ptr += m.w;
              printf("\n");
          }
          printf("------------------------\n");
      }
  }

};

runtime::Module NCNNRuntimeCreate(const String& symbol_name, const String& graph_json, 
    const Array<String>& const_names) {
  auto n = make_object<NCNNRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.NCNNRuntimeCreate").set_body_typed(NCNNRuntimeCreate);
TVM_REGISTER_GLOBAL("runtime.module.loadbinary_ncnn")
  .set_body_typed(JSONRuntimeBase::LoadFromBinary<NCNNRuntime>);
}
}
}
