
#include <tvm/relay/expr_functor.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include "../../utils.h"
#include "../codegen_json/codegen_json.h"

#include <fstream>


namespace tvm {
namespace relay {
namespace contrib {

class NCNNJSONSerializer: public backend::contrib::JSONSerializer {
  using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
  using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;

public:
  NCNNJSONSerializer(const std::string& symbol, const Expr& expr) : JSONSerializer(symbol, expr)
  {}

  /*!
   * \brief A series of operators that form a composite dense layer.
   */
  struct CompositeDenseNode {
    const CallNode* dense = nullptr;
    const CallNode* bias = nullptr;
    const CallNode* activation = nullptr; 
  };
  
  /*!
   * \brief A series of operators that form a composite conv2d layer.
   */
  struct CompositeConvNode {
    const CallNode* conv = nullptr;
    const CallNode* bias = nullptr;
    const CallNode* activation = nullptr; 
  };
  
  /*!
   * \brief Visit call nodes and generate appropriate JSON node.
   *
   * \param cn The current call node.
   * \return A list of graph entry nodes
   */ 
  std::vector<JSONGraphNodeEntry> VisitExpr_(const CallNode* cn) override {
    if (cn->op.as<OpNode>()) {
      return JSONSerializer::VisitExpr_(cn);
    }
    if (!cn->op.as<FunctionNode>()) {
      LOG(FATAL) << "NCNN JSON runtime does not support calls to " 
                 << cn->op->GetTypeKey();
    }
    auto fn = cn->op.as<FunctionNode>();
    auto comp = fn->GetAttr<String>(attr::kComposite);
    ICHECK(comp.defined()) << "NCNN Json runtime only supports composite functions.";
    const std::string name = comp.value();
    std::shared_ptr<JSONGraphNode> json_node;
    if (name == "ncnn.dense") {
      json_node = CreateCompositeDenseJSONNode(cn);
    } else if (name == "ncnn.conv2d") {
      json_node = CreateCompositeConvJSONNode(cn);
    }
    else {
      LOG(FATAL) << "Unrecognized NCNN pattern: " << name;
    }
    return AddNode(json_node, GetRef<Expr>(cn));
  }

private:
  /*!
   * \brief Extract convolution ndoes from a composite function.
   * 
   * \param cn The call node of the composite function.
   * \return Extracted composite convolution nodes.
   */
  static CompositeConvNode UnpackCompositeConvolution(const CallNode* cn) {
    CompositeConvNode nodes{};
    const auto* fn = cn->op.as<FunctionNode>();
    ICHECK(fn);

    // Traverse composite dense function from child to parent
    const auto* current_call = fn->body.as<CallNode>();
    if (backend::IsOp(current_call, "nn.relu")) {
      nodes.activation = current_call;
      current_call = current_call->args[0].as<CallNode>();
    }
    if (backend::IsOp(current_call, "nn.bias_add")) {
      nodes.bias = current_call;
      current_call = current_call->args[0].as<CallNode>();
    }
    // Enforce a dense node exists at this point during traversal
    ICHECK(backend::IsOp(current_call, "nn.conv2d"));
    nodes.conv = current_call;
    return nodes;
  }

   /*!
   * \brief Create a JSON representation of a composite convolution.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateCompositeConvJSONNode(const CallNode* cn) {
    CompositeConvNode nodes = UnpackCompositeConvolution(cn);

    std::string name = "nn.conv2d";

    // Inputs must be added in the same order they appear in the relay graph.
    std::vector<JSONGraphNodeEntry> inputs;
    inputs.push_back(VisitExpr(cn->args[0])[0]);
    inputs.push_back(VisitExpr(nodes.conv->args[1])[0]);
    
    if (nodes.bias) {
      inputs.push_back(VisitExpr(nodes.bias->args[1])[0]);
    }
    
    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, nodes.conv);
    if (nodes.activation) {
      std::vector<std::string> activation_type;
      if (backend::IsOp(nodes.activation, "nn.relu")) {
        activation_type = {"relu"};                   
      }
      std::vector<dmlc::any> act_attr;              
      act_attr.emplace_back(activation_type);      
      json_node->SetAttr("activation_type", act_attr);     
    }
    return json_node;
  }


  /*!
   * \brief Extract dense ndoes from a composite function.
   * 
   * \param cn The call node of the composite function.
   * \return Extracted composite dense nodes.
   */
  static CompositeDenseNode UnpackCompositeDense(const CallNode* cn) {
    CompositeDenseNode nodes{};
    const auto* fn = cn->op.as<FunctionNode>();
    ICHECK(fn);

    // Traverse composite dense function from child to parent
    const auto* current_call = fn->body.as<CallNode>();
    if (backend::IsOp(current_call, "nn.relu")) {
      nodes.activation = current_call;
      current_call = current_call->args[0].as<CallNode>();
    }
    if (backend::IsOp(current_call, "nn.bias_add")) {
      nodes.bias = current_call;
      current_call = current_call->args[0].as<CallNode>();
    }
    // Enforce a dense node exists at this point during traversal
    ICHECK(backend::IsOp(current_call, "nn.dense"));
    nodes.dense = current_call;
    return nodes;
  }

  /* \brief Create a JSON representation of a composite dense (fully-connected) operator.
   * 
   * \param cn The call to be represented
   * \return A JSON representation of a specific operator.
   */ 
  std::shared_ptr<JSONGraphNode> CreateCompositeDenseJSONNode(const CallNode* cn) {
    CompositeDenseNode nodes = UnpackCompositeDense(cn);
    std::string name = "nn.dense";

    // Input must be added in the same order they appear in the relay graph
    std::vector<JSONGraphNodeEntry> inputs;
    // input
    inputs.push_back(VisitExpr(cn->args[0])[0]);
    // weight for dense
    inputs.push_back(VisitExpr(nodes.dense->args[1])[0]);
    // bias for dense
    if (nodes.bias) {
      inputs.push_back(VisitExpr(nodes.bias->args[1])[0]);
    }
    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, nodes.dense);
    if (nodes.activation) {
      std::vector<std::string> activation_type;
      if (backend::IsOp(nodes.activation, "nn.relu")) {
        activation_type = {"relu"};
      }
      std::vector<dmlc::any> act_attr;
      act_attr.emplace_back(activation_type);
      json_node->SetAttr("activation_type", act_attr);
    }
    return json_node;
  }
};
/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module.
 */
runtime::Module NCNNCompiler(const ObjectRef& ref) {
  auto func = Downcast<Function>(ref);
  std::string func_name(tvm::relay::backend::GetExtSymbol(func));
  std::cout << "Building ncnn JSON subgraph: " << func_name << std::endl;
  Array<String> const_names = Array<String>();
  
  // Parse graph representation
  NCNNJSONSerializer serializer(func_name, func);
  serializer.serialize();
  std::string graph_json = serializer.GetJSON();
  
  // Generate runtime library 
  const auto* pf = runtime::Registry::Get("runtime.NCNNRuntimeCreate");
  ICHECK(pf != nullptr) << "Cannot find ncnn JSON runtime module to create";
  runtime::Module lib = (*pf)(func_name, graph_json, serializer.const_names());

  return lib;
}

TVM_REGISTER_GLOBAL("relay.ext.ncnn").set_body_typed(NCNNCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
