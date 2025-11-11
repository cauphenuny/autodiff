#include "util.hpp"

#include <initializer_list>
#include <vector>

template <typename T> class Storage {
    enum class Type { own, view };
    const Type _type;
    T* _data;

public:
    std::vector<size_t> strides;
    std::vector<size_t> offsets;

    Storage(const std::initializer_list<size_t>& shape)
        : _type(Type::own), offsets(shape.size(), 0) {
        strides.resize(shape.size());
        size_t size = 1, i = shape.size();
        for (auto it = shape.end(); it != shape.begin();) {
            --it;
            --i;
            strides[i] = size;
            size *= *it;
        }
        _data = new T[](size);
    }
    Storage(T* data, std::vector<size_t>&& strides, std::vector<size_t>&& offsets)
        : _type(Type::view), _data(data), strides(std::move(strides)),
          offsets(std::move(offsets)) {}
    ~Storage() {
        if (_type == Type::own) delete[] _data;
    }

    T* data() { return _data; }

    auto& operator[](this auto& self, std::vector<size_t> idx) {
        size_t index = 0;
        size_t i = 0;
        for (auto arg : idx) {
            index += (arg + self.offsets[i]) * self.strides[i];
            ++i;
        }
        return self._data[index];
    }
    auto& operator[](this auto self, size_t first, auto... args) {
        return self[std::vector{first, args...}];
    }
};

struct SliceRange {
    size_t start, end, step;
    SliceRange(size_t pos) : start(pos), end(pos), step(0) {}
    SliceRange(std::initializer_list<size_t> range)
        : start(*range.begin()), end(*(range.begin() + 1)), step(1) {
        if (range.size() == 3) step = *(range.begin() + 2);
    }
};

template <typename T> class Tensor {
    Storage<T> _storage;
    std::vector<size_t> _shape;
    size_t _size;
    struct IndexIterator {
        const std::vector<size_t>& shape;
        std::vector<size_t> idx;
        bool done;

        IndexIterator(const std::vector<size_t>& shape, bool done = false)
            : shape(shape), idx(shape.size(), 0), done(done) {
            if (shape.empty()) this->done = true;
        }
        std::vector<size_t> operator*() const { return idx; }
        IndexIterator& operator++() {
            for (int i = static_cast<int>(idx.size()) - 1; i >= 0; --i) {
                idx[i]++;
                if (idx[i] < shape[i]) return *this;
                idx[i] = 0;
            }
            done = true;
            return *this;
        }
        bool operator!=(const IndexIterator& other) const { return done != other.done; }
    };

    struct IndexRange {
        const std::vector<size_t>& shape;
        IndexRange(const std::vector<size_t>& shape) : shape(shape) {}
        IndexIterator begin() const { return IndexIterator(shape); }
        IndexIterator end() const { return IndexIterator(shape, true); }
    };

    IndexRange indexes() const { return IndexRange(_shape); }

public:
    auto& shape() { return _shape; }
    const auto& shape() const { return _shape; }
    const auto& storage() const { return _storage; }
    size_t size() const { return _size; }
    T* data() const { return _storage._data(); }

    Tensor(T value) : _storage({1}), _shape({1}), _size(1) { _storage[0] = value; }
    Tensor(Storage<T>&& storage, std::vector<size_t>&& shape)
        : _storage(std::move(storage)), _shape(std::move(shape)) {}
    Tensor(std::initializer_list<size_t> shape) : _storage(shape), _shape(shape) {}
    Tensor(std::initializer_list<size_t> shape, std::initializer_list<T> value)
        : _storage(shape), _shape(shape) {
        size_t i = 0;
        for (auto it = value.begin(); it != value.end(); ++it) {
            _storage[i] = *it;
            ++i;
        }
    }

    Tensor<T> operator=(const Tensor<T>& other) {
        if (this == &other) return *this;
        if (_shape != other._shape)
            runtimeError("shape not match: {} vs {}", _shape, other._shape);
        for (size_t i = 0; i < _shape.size(); ++i) {
            if (_shape[i] != other._shape[i])
                runtimeError("shape not match: {} vs {}", _shape, other._shape);
        }
        for (auto idx : this->indexes()) {
            this->_storage[idx] = other._storage[idx];
        }
        return *this;
    }

    bool operator==(const Tensor<T>& other) const {
        if (_shape != other._shape) return false;
        for (size_t i = 0; i < _shape.size(); ++i) {
            if (_shape[i] != other._shape[i]) return false;
        }
        for (auto idx : this->indexes()) {
            if (this->_storage[idx] != other._storage[idx]) return false;
        }
        return true;
    }

    // Tensor<T> initial_diff() const override {
    //     if (_shape.size() > 1) {
    //         runtimeError("can not backward tensor with multiple dimension");
    //     }
    //     return Tensor<T>(1);
    // }

    static Tensor<T> zeros(std::initializer_list<size_t> shape) {
        return Tensor<T>(shape);
    }

    static Tensor<T> ones(std::initializer_list<size_t> shape) {
        Tensor<T> t(shape);
        for (auto idx : t.indexes()) {
            t._storage[idx] = 1;
        }
        return t;
    }
};