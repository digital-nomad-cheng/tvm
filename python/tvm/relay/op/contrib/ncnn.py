# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument, dangerous-default-value
"""ncnn library supported operators."""
import tvm
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from ...dataflow_pattern import is_constant, is_op, wildcard
from .register import register_pattern_table 

def partition_for_ncnn(mod, params=None):
    """Partition the graph greedily offloading supported
    operators to ncnn.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.

    Returns
    -------
    ret : annotated and partitioned module.
    """
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.MergeComposite(ncnn_pattern_table()),
            transform.AnnotateTarget("ncnn"),
            transform.PartitionGraph(),
        ]
    )

    return seq(mod)

@register_pattern_table("ncnn")
def ncnn_pattern_table():
    """Get the ncnn pattern table"""
    def dense_pattern():
        """Create a dense (fully-connected) pattern.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution pattern.
        """
        pattern = is_op("nn.dense")(wildcard(), is_constant())
        pattern = pattern.optional(lambda x: is_op("nn.bias_add")(x, is_constant()))
        return pattern
    
    def check_dense(extractl):
        """Check dense pattern is supported by ncnn."""
        return True
    
    return [
        ("ncnn.dense", dense_pattern(), check_dense)
    ]
    
def _register_extern_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported by ncnn.

    Parameters
    ----------
    op_name : Str
        The name of the operator that will be registered.
    Returns
    -------
    f: callable
        A function that returns if the operator is supported by ncnn.
    """

    @tvm.ir.register_op_attr(op_name, "target.ncnn")
    def _func_wrapped(expr):
        return supported

    return _func_wrapped 

# _register_extern_op_helper("nn.dense")
_register_extern_op_helper("reshape")
