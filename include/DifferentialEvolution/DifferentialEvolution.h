#ifndef DIFFERENTIAL_EVOLUTION_H
#define DIFFERENTIAL_EVOLUTION_H

#include <random>
#include <cassert>
#include <vector>
#include <iostream>

namespace DifferentialEvolution{

    class Individual{
    private:
        std::vector<double> m_c;
        bool m_requires_eval;
        double m_cost;
    public:
        Individual(const std::vector<double> &c) : m_c(c), m_requires_eval(true), m_cost(1e30) {
            assert(c.size() > 0);
        };
        /// Set the coefficients in this individual
        void set_coeff(const std::vector<double> &c){
            m_c = c;
            m_requires_eval = true;
            assert(c.size() > 0);
        };
        /// Get the coefficients of this individual
        const std::vector<double> &get_coeff () const { return m_c; };
        /// Return true if it needs to be evaluated
        bool requires_eval(){ return m_requires_eval; }
        /// Set the cost of this individual
        void set_cost(double cost){ m_cost = cost; m_requires_eval = false; assert(requires_eval() == false);}
        /// Get the cost of this individual
        double get_cost() const { return m_cost; };
    };

    class DifferentialEvolutionConfiguration{
    public:
        std::vector<std::pair<double,double> > bounds;
        std::size_t popsize;
        double F = 1; ///< Differential weight
        double CR = 0.5; ///< Crossover probability
        double epsilon_ftol = -1e30; ///< If the functional value is less than this value, we terminate
    };

    typedef std::vector<std::pair<double, double> > bounds_type;

    class DifferentialEvolutionOptimizer{
    private:
        DifferentialEvolutionConfiguration config;
        std::function<double(const std::vector<double> &c)> f;
        std::random_device rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 gen; // Standard mersenne_twister_engine
        std::vector<Individual> population;
        std::size_t m_generation_counter;
    public:

        DifferentialEvolutionOptimizer(const std::function<double(const std::vector<double> &c)> &f, bounds_type bounds, std::size_t popsize) : f(f), gen(rd()), m_generation_counter(0)
        {
            config.bounds = bounds;
            config.popsize = popsize;
        };
        /// Generate the initial population of individuals
        void initialize(){
            // Loop over the individuals in the population
            for (auto iind = 0; iind < config.popsize; ++iind){
                
                // Generate the coefficients for this individual based on the bounds specified
                std::vector<double> v;
                for(auto &lb_ub : config.bounds){
                    v.push_back(std::uniform_real_distribution<>(lb_ub.first, lb_ub.second)(gen));
                }
                this->population.push_back(Individual(v));
            }
            // Evaluate the costs for each individual
            for (auto &ind: this->population){
                ind.set_cost(this->f(ind.get_coeff()));
            }
        }
        /// Return a reference to the config
        DifferentialEvolutionConfiguration & get_config(){ return config; };
        /// Get N unique indices
        std::vector<std::size_t> get_N_unique(std::size_t Nindices, std::size_t N){
            assert(N <= Nindices);
            // Short circuit if you want as many indices as the length
            if (N == Nindices){
                std::vector<std::size_t> out(N);
                for (auto i = 0; i < N; ++i){ out[i] = i; }
                return out;
            }
            // Otherwise, find N unique indices, each  >= 0 and <= Nindices
            std::uniform_int_distribution<> dis(0, Nindices);
            std::vector<std::size_t> indices;
            for (auto i = 0; i < N; ++i){
                while (true){
                    // Trial index
                    auto j = dis(gen);
                    if (std::find(indices.begin(), indices.end(), j) != indices.end()){
                        // It's already in the list of indices to keep; keep trying
                        continue;
                    }
                    else{
                        // Not being used yet; keep it
                        indices.push_back(j);
                        break;
                    }
                }
            }
            return indices;
        }
        /* 
         * Return a hybrid individual from the original and three
         * other individuals
         */
        Individual new_ind(const Individual &orig, const Individual &ind1, const Individual &ind2, const Individual &ind3, std::size_t R){
            std::vector<double> vn = orig.get_coeff();
            auto Ncoeff = vn.size();
            for (auto i = 0; i < Ncoeff; ++i){
                if (i == R || std::uniform_real_distribution<>(0, 1)(gen) < config.CR){
                    vn[i] = ind1.get_coeff()[i] + config.F*(ind2.get_coeff()[i] - ind3.get_coeff()[i]);
                }
            }
            return Individual(vn);
        }
        // Do one generation
        void do_generation(){
            auto Ncoeff = this->population[0].get_coeff().size();
            for (auto counter = 0; counter < config.popsize; ++counter){
                // Get a random individual (original) and three others
                // (actually their indices)
                auto uniques = this->get_N_unique(config.popsize-1, 4);

                // Get a random index
                auto R = std::uniform_int_distribution<>(0, Ncoeff-1)(gen);

                // Get the candidate individual
                Individual cand = this->new_ind(this->population[uniques[0]],
                                                this->population[uniques[1]],
                                                this->population[uniques[2]],
                                                this->population[uniques[3]],
                                                R);
                cand.set_cost(this->f(cand.get_coeff()));
                
                // If the original is worse than the candidate, swap
                if (this->population[uniques[0]].get_cost() > cand.get_cost()){
                    std::swap(this->population[uniques[0]], cand);
                }
            }
            // Sort the individuals in increasing cost
            std::sort(this->population.begin(),
                      this->population.end(),
                      [](const Individual &ind1, const Individual &ind2){
                          return ind1.get_cost() < ind2.get_cost();
                      });
            
            std::cout << m_generation_counter << " " << this->population.front().get_cost() << std::endl;
            // Increment the generation counter
            m_generation_counter++;
        };
        /// Optimize, with up to Ngen generations
        void optimize(std::size_t Ngen){
            this->initialize();
            for (auto igen = 0; igen < Ngen; ++igen){
                this->do_generation();
                if (this->population.front().get_cost() < config.epsilon_ftol){
                    return;
                }
            }
        };
    };
};


#endif