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
    SetupConstants(consts);
    BuildEngine();
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

private:
  /*!
   * \brief Build ncnn layer from JSON representation and cache
   * \note For the time being only one layer or operator is supported
   * per engine
   */
  void BuildEngine() {
    bool found_kernel_node = false;
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (found_kernel_node) {
        LOG(FATAL) 
          << "ncnn runtime module only supports one kernel node per function.";
      }
      if (node.GetOpType() == "kernel") {
        found_kernel_node = true;
        auto op_name = node.GetOpName();
        if (op_name == "nn.dense") {
          CreateInnerProductLayer(&layer_, node);
        } else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
      }
    }
  }
  
  /*!
   * \brief ncnn objects that we cache in order to avoid needing to construct 
   * a new layer each time.
   */
  struct CachedLayer {
    ncnn::Layer* op;
    ncnn::Option opt;
    ncnn::Mat in;
    ncnn::Mat out;
  };
  
  void CreateInnerProductLayer(CachedLayer* layer, const JSONGraphNode& node) {
    ncnn::Layer* op = ncnn::create_layer("InnerProduct");
    // collect inputs from json representation 
    std::vector<JSONGraphNodeEntry> inputs = node.GetInputs();
    size_t num_inputs= inputs.size();
    LOG(INFO) << "num_inputs parsed from ncnn json: " << num_inputs;
    bool has_bias;
    ICHECK(num_inputs >= 2U && num_inputs <= 3U)
      << "InnerProduct(dense) layer requires 3 inputs with a bias, 2 inputs without.";
    has_bias = num_inputs == 3;
    ncnn::Option opt;
    opt.num_threads = 2; // TODO: how to get num threads to use from tvm
    ncnn::ParamDict pd;
    if (has_bias) {
      pd.set(1, 1); // has bias
    } else {
      pd.set(1, 0); // has no bias
    }
    for (size_t i = 0; i < inputs.size(); i++) {
      auto tensor = inputs[i];
      JSONGraphNode node = nodes_[tensor.id_];
      if (node.GetOpType() == "const") {
        LOG(INFO) << i << "th node is " << "const/weight node";
        void* node_data = nullptr;
        int64_t* node_shape = nullptr;
        node_shape = data_entry_[EntryID(tensor)]->shape;
        LOG(INFO) << "shape of data is" << &node_shape;

      }
    }
    // pd.set(2, ...) // TODO: set weight size
    layer->op = op;
    op->load_param(pd); // load param/model structure
     
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

  CachedLayer layer_;
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
