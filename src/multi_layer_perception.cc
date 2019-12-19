/*
* Author: Ke Dong(kedong0810@gmail.com)
* Date: 2019-11-21
* Brief: customized mlp library for fast online inference
*/

#include "multi_layer_perception.h"

using namespace std;

namespace network_mlp{

bool Layer::init(nlohmann::json& js){
    input_dim_ = js["input_dim"].get<int>();
    output_dim_ = js["output_dim"].get<int>();
    act_func_ = js["activation"].get<string>();
    weight_.resize(input_dim_, output_dim_);
    bias_.resize(output_dim_, 1);

    // read weight and bias
    std::vector<std::vector<double>> weight_vec = js["weight"];
    std::vector<double> bias_vec = js["bias"];
    if (bias_vec.size() != weight_vec[0].size() or weight_vec.size() != input_dim_){
        cout << "size does not match" << endl;
        cout << "bias size: " << bias_vec.size() << " weight size: " << weight_vec.size() << endl;
        cout << "input size: " << input_dim_ << endl;
        return false;
    }
    for (int i = 0; i < weight_vec.size(); ++i){
        for (int j = 0; j < weight_vec[i].size(); ++j){
            weight_(i, j) =  weight_vec[i][j];
        }
    }
    for (int i = 0; i < output_dim_; ++i){
        bias_(i, 0) = bias_vec[i];
    }
    return true;
}

void Layer::relu(Eigen::MatrixXd& X){
    for (int i = 0; i < X.rows(); ++i){
        for (int j = 0; j < X.cols(); ++j){
            if (X(i, j) <= 0){
                X(i, j) = 0;
            }
        }
    }
}

bool Layer::forward(const Eigen::MatrixXd & input_vec, Eigen::MatrixXd & output_vec){
    output_vec = weight_.transpose() * input_vec + bias_;
    if (act_func_ == "relu"){
        relu(output_vec);
    }else if (act_func_ == "linear"){
        ;
    }else{
        cout << "not matched activation function: " << act_func_ << endl;
        return false;
    }
    return true;
}

Eigen::MatrixXd Layer::getWeight(){
    return weight_;
}

Eigen::MatrixXd Layer::getBias(){
    return bias_;
}

bool Network::init(ifstream& ins){
    nlohmann::json js;
    ins >> js;
    ins.clear();
    ins.seekg(0);

    vector<string> names_vec = js["name_list"];
    names_ = names_vec;
    for (int i = 0; i < names_vec.size(); ++i){
        string name = names_vec[i];
        Layer layer;
        if (!layer.init(js[name])){
            cout << "network initialization fails on layer: "<< name << endl;
            return false;
        }
        //cout << "layer: " << name << " initializes successfully" << endl;
        layers_.push_back(layer);
    }
    input_dim_ = js["input_dim"];
    output_dim_ = js["output_dim"];

    input_vec_.resize(input_dim_, 1);

    return true;
}

bool Network::forward(const std::vector<double> & input_vals, std::vector<double> & output_vals){
    if (input_vals.size() != input_dim_){
        cout << "input size does not match: " << input_vals.size() << " != " << input_dim_ << endl;
        return false;
    }
    for (int i = 0; i < input_dim_; ++i){
        input_vec_(i, 0) = input_vals[i];
    }

    inputs = input_vec_;
    for (int i = 0; i < layers_.size(); ++i ){
        layers_[i].forward(inputs, output_vec_);
        inputs = output_vec_;
    };

    if (output_vec_.rows() != output_dim_ or output_vec_.cols() != 1){
        cout << "output dimension does not match: " << output_vec_.rows() << "x" << output_vec_.cols() << endl;
        return false;
    }

    if (output_vals.size() != output_dim_){
        output_vals.resize(output_dim_, 0);
    }
    for (int i = 0; i < output_dim_; ++i){
        output_vals[i] = output_vec_(i, 0);
    }
    return true;
}

void Network::networkInfoPrinter(){
    cout << "================= Network Information==========================" << endl;
    cout << "---- Input dimension: " << input_dim_ << endl;
    cout << "---- Output dimension:" << output_dim_ << endl;
    for (int i = 0; i < layers_.size(); ++i){
        cout << "---- " << i << "-th layer: " << endl;
        cout << "WEIGHT: " << endl;
        cout << layers_[i].getWeight() << endl;
        cout << "BIAS: " << endl;
        cout << layers_[i].getBias() << endl;
        cout << endl;
    }
}

} // namespace network_mlp
