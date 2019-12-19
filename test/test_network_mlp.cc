#include "multi_layer_perception.h"

#include <fstream>
#include <sstream>
#include <string.h>

using namespace std;
using namespace network_mlp;

int TextReader(const char* filename, int numSkipLine, int num_dim, std::vector<std::vector<double>>& database){
    // use stringstream to parse lines
    ifstream input_file;
    string line;
    input_file.open(filename);
    double val;
    int count = 0;
    if (input_file.is_open()){
        for (int i = 0; i < numSkipLine; ++i){
            getline(input_file, line);
        }
        while(getline(input_file, line)){
            stringstream ss(line);
            vector<double> point;
            while(ss >> val){
                point.push_back(val);
            }
            database.push_back(point);
            count += 1;
        }
    }
    return count;
}


int main(int argc, char** argv){
    char* network_info_pth = argv[1];
    char* data_file_name = argv[2];
    int num_skip_line = 1;
    int num_dim = 4;
    std::vector<std::vector<double>> database;

    int num_dataset = TextReader(data_file_name, num_skip_line, num_dim, database);

    ifstream ins;
    ins.open(string(network_info_pth));

    Network network;
    if (!network.init(ins)){
        return 0;
    }
    ins.close();
    //network.networkInfoPrinter();
    
    std::vector<double> output;
    std::vector<double> input = {0, 0};
    std::vector<double> result; 
    int r = 3;
    for (int i = 0; i < num_dataset - r; ++i){
        std::vector<double> point;
        input[0] = database[i][1] - database[i][3]; 
        input[1] = database[i+r][3] - database[i][3];
        network.forward(input, output);
        result.push_back(output[0]);
    }

    
    std::vector<double> time, original_desired, input_1, input_2, output_1, modified;
    for (int i = 0; i < num_dataset - r; ++i){
        time.push_back(database[i][0]);
        original_desired.push_back(database[i][3]);
        input_1.push_back(database[i][1] - database[i][3]);
        input_2.push_back(database[i+r][3] - database[i][3]);
        output_1.push_back(database[i+1][3] - database[i][3]);
        modified.push_back(result[i] + database[i][3]);
    }

    cout << "test successfully" << endl;

    return 1;
}