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

    @tvm.ir.register_op_attr(op_name, "target.ncnn"):
    def _func_wrapped(expr):
        return supported

    return _func_wrapped 

_register_extern_op_helper("nn.dense")

