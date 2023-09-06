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

if(USE_NCNN_CODEGEN STREQUAL "ON")
		tvm_file_glob(GLOB ncnn_RELAY_CONTRIB_SRC src/relay/backend/contrib/ncnn/*)
		list(APPEND COMPILER_SRCS ${ncnn_RELAY_CONTRIB_SRC})
		tvm_file_glob(GLOB ncnn_CONTRIB_SRC src/runtime/contrib/ncnn/ncnn_runtime.cc src/runtime/contrib/ncnn/ncnn_runtime.h)
		list(APPEND RUNTIME_SRCS ${ncnn_CONTRIB_SRC})
    
    set(ncnn_PATH "/opt/ncnn/build/install")
    tvm_file_glob(GLOB ncnn_CONTRIB_SRC src/runtime/contrib/ncnn/*)
    set(ncnn_INCLUDE_DIRS ${ncnn_PATH}/include)
    include_directories(${ncnn_INCLUDE_DIRS})
    # message(STATUS "ncnn include:" ${ncnn_INCLUDE_DIRS})  
    find_library(EXTERN_ncnn_LIB 
      NAMES ncnn
      HINTS "${ncnn_PATH}" "${ncnn_PATH}/lib"
    )
    set(ncnn_DIR "/opt/ncnn/build/install/lib/cmake/ncnn")
    find_package(ncnn REQUIRED) 
    list(APPEND TVM_RUNTIME_LINKER_LIBS ncnn)# ${EXTERN_ncnn_LIB})
    list(APPEND RUNTIME_SRCS ${ncnn_CONTRIB_SRC})
    message(STATUS "Build with ncnn graph executor support...")
endif()
	
	



