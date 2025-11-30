#include <cmath>
#include <string>
#include <set>
#include <functional>
#include <iostream>

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

    void backward() {
        std::vector<Value*> topo;
        std::set<Value*> visited;

        std::function<void(Value*)> fn_topo = [&](Value* v) {
            if (visited.find(v) == visited.end()) {
                visited.insert(v);
                for (auto child: v->_prev) {
                    fn_topo(child);
                }
                topo.push_back(v);
            }
        };

        fn_topo(this);

        this->grad = 1.0;
        for (auto v: topo) {
            v->_backward();
        }
    }

    Value* operator-() {
        return (*this) * -1;
    }

    Value* operator-(Value& other) {
        return (*this) + (*(-other));
    }

    Value* operator/(Value& other) {
        return (*this) * (*(other.pow(-1)));
    }
};

// C++ equivalent of Python's __repr__ - allows printing with std::cout << value
std::ostream& operator<<(std::ostream& os, const Value& v) {
    os << "Value(data=" << v.data << ", grad=" << v.grad << ")";
    return os;
}

Value* operator+(int self, Value& other) {
    Value* self_val = new Value(self);
    return self_val->operator+(other);
}

Value* operator-(int self, Value& other) {
    Value* self_val = new Value(self);
    return self_val->operator-(other);
}

Value* operator*(int self, Value& other) {
    Value* self_val = new Value(self);
    return self_val->operator*(other);
}

Value* operator/(int self, Value& other) {
    Value* self_val = new Value(self);
    return self_val->operator/(other);
}
