
#include "DifferentialEvolution/DifferentialEvolution.h"

double Griewangk(const std::vector<double> &x){
    double sum1 = 0.0;
    double prod1 = 1.0;
    for (auto i = 0; i < x.size(); ++i){
        sum1 += pow(x[i], 2);
        prod1 *= cos(x[i]/sqrt(i+1));
    }
    auto f = sum1/4000.0 - prod1 + 1;
    return f;
}

using namespace DifferentialEvolution;

int main(){
    for (std::size_t D: {20})//,20,30,40})
    {
        bounds_type bounds;
        for (auto i = 0; i < D; ++i){
            bounds.push_back(std::make_pair(-10,10));
        }

        DifferentialEvolutionOptimizer de(Griewangk, bounds, D*5);
        auto &config = de.get_config();
        config.epsilon_ftol = 1e-6;
        de.optimize(12000);
    }
    return EXIT_SUCCESS;
}