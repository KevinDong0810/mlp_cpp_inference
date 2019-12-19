# A light-weight multi-layer perception library for fast online inference
This library is designed for scenarios where you want to deploy mlp networks trained by tensorflow or pytorch into embedded systems in a c++ environment, e.g. manipulators, quadrotors. It can read weight and bias parameters stored in json files, and achieves fast online inference speed

## Install and usage
The library consists of a header file in ``include/multi_layer_perception.h`` and a c++ file in ``nulti_layer_perception``. To read json files, a header file in ``include/json.hpp`` is also included. There is only one extra package you need to install: [``Eigen``](https://eigen.tuxfamily.org/dox/index.html). To run the program, you just need incorporate the above code into your program. 

## Json file configuration
An example of json file is presented in ``models/example.json``. Currently, only ``relu`` and ``linear`` activation functions are supported. But other activation functions can be easily added in ``include/multi_layer_perception.h`` -> ``Layer``

## Test
    ```
    make test
    cd build
    ./test_network ../models/mlp_rb_x.json ../test/test_data.txt
    ```