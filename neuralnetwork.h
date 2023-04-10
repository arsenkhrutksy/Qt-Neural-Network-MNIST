#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H


#include <vector>

class neuralnetwork
{
public:
    neuralnetwork(int, int, int, float);
    ~neuralnetwork();

    int inodes;
    int hnodes;
    int onodes;

    float lrate;

    std::vector<std::vector <long double>> weights_ih;
    std::vector<std::vector <long double>> weights_ho;

    std::vector<long double> query(std::vector<long double> input_list);
    void train(std::vector<long double>, std::vector<long double>);

    void set_weights(std::vector<std::vector<long double>>,std::vector<std::vector<long double>>);
};
#endif // NEURALNETWORK_H
