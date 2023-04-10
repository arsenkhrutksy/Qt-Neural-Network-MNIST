#include "neuralnetwork.h"

#include <random>
#include <math.h>
#include <vector>
#include <iostream>



std::vector<long double> dot(std::vector< std::vector<long double>> m1, std::vector<long double> m2) {
    if (m1[0].size() != m2.size()) { std::cout << m1[0].size() << ' '  << m2.size() << "\nerr\n";  return {}; }
    std::vector<long double> outm;
    long double temp;
    for (int i = 0; i < m1.size(); i++) {
        temp = 0;
        for (int j = 0; j < m2.size(); j++) temp += m1[i][j] * m2[j];
        outm.push_back(temp);
    }
    return outm;
}

std::vector<long double> t_dot(std::vector< std::vector<long double>> m1, std::vector<long double> m2) {
    if (m1.size() != m2.size()) { std::cout << m1.size() << ' ' << m2.size() << "\nerr\n";  return {}; }
    std::vector<long double> outm;
    long double temp;
    for (int i = 0; i < m1[0].size(); i++) {
        temp = 0;
        for (int j = 0; j < m2.size(); j++) temp += m1[j][i] * m2[j];
        outm.push_back(temp);
    }
    return outm;
}

std::vector <long double> n_expit(std::vector <long double> db) {
    for (auto& x : db) {
        x = 1 / (1 + exp(0 - x));
    }
    return db;
}

std::vector <long double> _sub(std::vector <long double> m1, std::vector <long double> m2) {
    if (m1.size() != m2.size()) {
        return {};
    }
    std::vector <long double> out;
    for (int i = 0; i < m1.size(); i++) out.push_back(m1[i] - m2[i]);
    return out;
}


neuralnetwork::neuralnetwork(int input_nodes, int hidden_nodes, int output_nodes, float learning_rate) {

    inodes = input_nodes ;
    hnodes = hidden_nodes ;
    onodes = output_nodes ;
    lrate = learning_rate;


    std::default_random_engine generator;
    std::normal_distribution<long double> dist_ih(0.0, std::pow(inodes, -0.5));
    std::normal_distribution<long double> dist_oh(0.0, std::pow(hnodes, -0.5));


    std::vector <long double> temp;
    for (int i = 0; i < hnodes; i++) {
        for (int j = 0; j < inodes; j++) { temp.push_back(dist_ih(generator)); }
        weights_ih.push_back(temp);
        temp.clear();
    }

    for (int i = 0; i < onodes; i++) {
        for (int j = 0; j < hnodes; j++) { temp.push_back( dist_oh(generator)); }
        weights_ho.push_back(temp);
        temp.clear();
    }

}

neuralnetwork::~neuralnetwork() {

}

std::vector<long double> neuralnetwork::query(std::vector<long double> input_list) {
    std::vector<long double> hidden_inputs = dot(weights_ih, input_list);
    std::vector<long double> hidden_outputs = n_expit(hidden_inputs);
    std::vector<long double> final_inputs = dot(weights_ho, hidden_outputs);
    std::vector<long double> final_outputs = n_expit(final_inputs);
    return final_outputs;
}

void neuralnetwork::train(std::vector<long double> input_list, std::vector<long double> target_list) {


    std::vector<long double> hidden_inputs = dot(weights_ih, input_list);

    std::vector<long double> hidden_outputs = n_expit(hidden_inputs);

    std::vector<long double> final_inputs = dot(weights_ho, hidden_outputs);

    std::vector<long double> final_outputs = n_expit(final_inputs);

    std::vector<long double> output_errors = _sub(target_list, final_outputs);

    std::vector<long double> hidden_errors = t_dot(weights_ho, output_errors);


    std::vector<long double> temp1, temp2;
    for (int i = 0; i < output_errors.size(); i++) {
        temp1.push_back((output_errors[i] * final_outputs[i] * (1.0 - final_outputs[i])));
    }
    for (int i = 0; i < hidden_errors.size(); i++) {
        temp2.push_back((hidden_errors[i] * hidden_outputs[i] * (1.0 - hidden_outputs[i])));
    }

    std::vector<std::vector<long double>> who_change, wih_change;
    std::vector<long double> temp_change;


    for(int i = 0; i < temp1.size(); i++){
        for (int j = 0; j < hidden_outputs.size(); j++) {
            temp_change.push_back(temp1[i] * hidden_outputs[j]);
        }
        who_change.push_back(temp_change);
        temp_change.clear();
    }
    for (int i = 0; i < temp2.size(); i++) {
        for (int j = 0; j < input_list.size(); j++) {
            temp_change.push_back(temp2[i] * input_list[j]);
        }
        wih_change.push_back(temp_change);
        temp_change.clear();
    }

    for (int i = 0; i < weights_ho.size(); i++) {
        for (int j = 0; j < weights_ho[0].size(); j++) {
            weights_ho[i][j] += lrate * who_change[i][j];
        }
    }

    for (int i = 0; i < weights_ih.size(); i++) {
        for (int j = 0; j < weights_ih[0].size(); j++) {
            weights_ih[i][j] += lrate * wih_change[i][j];
        }
    }


}

void neuralnetwork::set_weights(std::vector<std::vector<long double> > wih, std::vector<std::vector<long double> > who){
    weights_ih = wih;
    weights_ho = who;
}
