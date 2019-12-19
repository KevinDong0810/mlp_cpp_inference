/*
* Author: Ke Dong(kedong0810@gmail.com)
* Date: 2019-11-21
* Brief: customized mlp library for fast online inference
*/

#ifndef MULTI_LAYER_PERCEPTION_H
#define MULTI_LAYER_PERCEPTION_H

#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <iostream>

#include <Eigen/Dense>
#include "json.hpp"

namespace network_mlp{

/* A Class for a neuron layer
*/
class Layer{

public:
    Layer(){}
    ~Layer(){}

    /** Loads weight and bias parameters from a given json object
     *  \param js:  a json object that contains weights and bias
     * 
     *  \return true or false
     */ 
    bool init(nlohmann::json& js);

    /** Does an online forward inference
     *  \param input_vec: an input vector from previous layers
     *  \param output_vec: a vector to be populated with layer outputs
     * 
     *  \return false if anything wrong 
     */
    bool forward(const Eigen::MatrixXd & input_vec, Eigen::MatrixXd & output_vec);

    /** Return layer weights
     */ 
    Eigen::MatrixXd getWeight();

    /** Return layer bias
     */     
    Eigen::MatrixXd getBias();    

private:
    int input_dim_;
    int output_dim_;
    std::string act_func_;  // indicate which activation functions to be used, only support 'relu' or 'linear' now
    Eigen::MatrixXd weight_;
    Eigen::MatrixXd bias_;

    void relu(Eigen::MatrixXd& X);
};

/* A Class for a neuron network that consists of several layers
*/

class Network{
public:
    Network(){}
    ~Network(){}

    /** Initializes the network with a configuration file
     *  \param ins: a ifstream pointing to the configuration file
     * 
     *  \return true if initialization succeeds
     */  
    bool init(std::ifstream& ins);

    /** Does a forward passing
     *  \param input_vals: a vector containing inputs
     *  \param output_vals: a vector to be populated with inference results
     *  
     *  \return false if anything wrong
     */ 
    bool forward(const std::vector<double> & input_vals, std::vector<double> & output_vals );

    /** Prints the network information for debugging
     */ 
    void networkInfoPrinter();

private:
    std::vector<Layer> layers_;
    std::vector<std::string> names_;
    Eigen::MatrixXd input_vec_;
    Eigen::MatrixXd inputs;
    Eigen::MatrixXd output_vec_;
    int input_dim_;
    int output_dim_;
};

} // namespace network_mlp

#endif