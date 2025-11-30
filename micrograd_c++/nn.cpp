#include "engine.cpp"
#include <random>
#include <vector>
#include <iostream>

class Module {
    public:
        void zero_grad() {
            for (auto p: this->parameters()) {
                p->grad = 0;
            }
        }

        std::vector<Value*> parameters() {
            return {};
        }
};

class Neuron: public Module {
    public:
        std::vector<Value> w;
        Value b;
        bool nonlin;
        
        Neuron(int nin, bool nonlin = true) : b(0), nonlin(nonlin) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(-1.0, 1.0);
            
            for (int i = 0; i < nin; i++) {
                w.push_back(Value(dis(gen)));
            }
        }

        Value operator()(std::vector<Value>& x) {
            Value* act = new Value(b);
            for (size_t i = 0; i < w.size(); i++) {
                Value* product = w[i] * x[i];
                Value* sum = *act + *product;
                delete act;
                delete product;
                act = sum;
            }
            
            if (nonlin) {
                Value* relu_result = act->relu();
                Value ret = *relu_result;
                delete relu_result;
                delete act;
                return ret;
            } else {
                Value ret = *act;
                delete act;
                return ret;
            }
        }

        std::vector<Value*> parameters() {
            std::vector<Value*> params;
            for (auto& wi: w) {
                params.push_back(&wi);
            }
            params.push_back(&b);
            return params;
        }
};

class Layer: public Module {
    public: 
        std::vector<Neuron> neurons;

        Layer(int nin, int nout, bool nonlin = true) {
            neurons.reserve(nout);
            for (int i = 0; i < nout; i++) {
                neurons.emplace_back(nin, nonlin);
            }
        }

        std::vector<Value> operator()(std::vector<Value>& x) {
            std::vector<Value> out;
            for (auto& neuron: neurons) {
                out.push_back(neuron(x));
            }
            return out;
        }
        
        // Helper to get single value when layer has 1 neuron (matching Python behavior)
        Value single(std::vector<Value>& x) {
            return neurons[0](x);
        }

        std::vector<Value*> parameters() {
            std::vector<Value*> params;
            for (auto& neuron: neurons) {
                for (auto& param: neuron.parameters()) {
                    params.push_back(param);
                }
            }
            return params;
        }
};

class MLP: public Module {
    public:
        std::vector<Layer> layers;

        MLP(int nin, std::vector<int> nouts) {
            std::vector<int> sz;
            sz.push_back(nin);
            for (int nout : nouts) {
                sz.push_back(nout);
            }
            
            for (size_t i = 0; i < nouts.size(); i++) {
                bool nonlin = (i != nouts.size() - 1);
                layers.push_back(Layer(sz[i], sz[i + 1], nonlin));
            }
        }
        
        Value operator()(std::vector<Value>& x) {
            std::vector<Value> current = x;
            for (auto& layer : layers) {
                current = layer(current);
            }
            return current[0];
        }
        
        std::vector<Value*> parameters() {
            std::vector<Value*> params;
            for (auto& layer : layers) {
                for (auto& param : layer.parameters()) {
                    params.push_back(param);
                }
            }
            return params;
        }
};

std::ostream& operator<<(std::ostream& os, const Neuron& n) {
    os << "Neuron(";
    for (size_t i = 0; i < n.w.size(); i++) {
        if (i > 0) os << ", ";
        os << n.w[i];
    }
    os << ", " << n.b << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const Layer& l) {
    os << "Layer(";
    for (size_t i = 0; i < l.neurons.size(); i++) {
        if (i > 0) os << ", ";
        os << l.neurons[i];
    }
    os << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const MLP& m) {
    os << "MLP(";
    for (size_t i = 0; i < m.layers.size(); i++) {
        if (i > 0) os << ", ";
        os << m.layers[i];
    }
    os << ")";
    return os;
}

int main() {
    std::vector<Value> x = {Value(2.0), Value(3.0), Value(-1.0)};
    MLP n(3, {4, 4, 1});
    
    std::vector<std::vector<Value>> xs = {
        {Value(2.0), Value(3.0), Value(-1.0)},
        {Value(3.0), Value(-1.0), Value(0.5)},
        {Value(0.5), Value(1.0), Value(1.0)},
        {Value(1.0), Value(1.0), Value(-1.0)}
    };
    std::vector<double> ys = {1.0, -1.0, -1.0, 1.0};
    
    std::vector<Value> ypred;
    for (auto& xi : xs) {
        ypred.push_back(n(xi));
    }

    std::cout << "[";
    for (size_t i = 0; i < ypred.size(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << ypred[i];
    }
    std::cout << "]" << std::endl;

    std::cout << "[";
    for (size_t i = 0; i < ys.size(); i++) {
        if (i > 0) std::cout << ", ";
        Value ygt_val(ys[i]);
        Value* diff = ypred[i] - ygt_val;
        Value* squared = diff->pow(2);
        std::cout << *squared;
        delete diff;
        delete squared;
    }
    std::cout << "]" << std::endl;
    
    return 0;
}

