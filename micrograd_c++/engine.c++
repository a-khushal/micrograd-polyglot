#include <iostream>
#include <cmath>
#include <string>
#include <set>
#include <functional>

class Value {
    public:
        double mutable data;
        double mutable grad;
        std::set<Value*> _prev;
        std::string _op;
        std::function<void()> _backward;

    Value(double data,
          std::set<Value*> prev = {},
          std::string op = ""
    ) {
        this->data = data;
        this->_prev = prev;
        this->_op = op;
        this->grad = 0.0;
        this->_backward = []() {};
    }

    Value* operator+(Value& other) {
        Value* out = new Value(this->data + other.data, {this, &other}, "+");
    
        out->_backward = [this, &other, out]() {  // Capture other by reference
            this->grad += out->grad;
            other.grad += out->grad;
        };
    
        return out;
    }

    Value* operator+(int other) {
        Value* other_val = new Value(other);
        return operator+(*other_val);
    }

    Value* operator*(Value& other) {
        Value* out = new Value(this->data * other.data, {this, &other}, "*");

        out->_backward = [this, &other, out]() {
            this->grad += other.data * out->grad;
            other.grad += this->data * out->grad;
        };

        return out;
    }

    Value* operator*(int other) {
        Value* other_val = new Value(other);
        return operator*(*other_val);
    }

    Value* pow(int other) {
        Value* out = new Value(std::pow(this->data, other), {this}, "**" + std::to_string(other));

        out->_backward = [this, &other, out]() {
            this->grad += other * std::pow(this->data, other-1) * out->grad;
        };

        return out;
    }

    Value* relu() {
        Value* out = new Value(std::max(0.0, this->data), {this}, "ReLU");

        out->_backward = [this, out]() {
            this->grad += (out->data > 0) * out->grad;
        };

        return out;
    }
};

int main() {
    Value a = Value(2.0);
    int b = 3.0;
    Value* c = a * b;
    std::cout << c->data << std::endl;
    return 0;
}
