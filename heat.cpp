#include <cmath>
#include <iostream>
#include <initializer_list>
#include <memory>
#include <set>
#include <map>
#include <stdexcept>
#include <utility>
#include <typeinfo>
#include <functional>
#include <numeric> // std::inner_product
#include <iterator>

// ------------------------------------------------------------ //
// common utilities
template <typename T, typename U>
void sizeCheck(const Vector<T>& lhs, const Vector<T>& rhs) {
    if(lhs.len() == 0 || rhs.len() == 0 || lhs.len() != rhs.len()) {
        throw std::logic_error("vector length does not fit."); 
    }
}

template <typename T, typename U>
void sizeCheck(const Matrix<T>& lhs, const Vector<T>& rhs) {
    if(lhs.empty() || rhs.empty())
        throw std::logic_error("empty candidate(s)");

    auto matSize = lhs.size();
    if (matSize.second != rhs.len())
        throw std::logic_error("vector and matrix size not compatible");
}

// ------------------------------------------------------------ //
// class Vector 
template <typename T>
class Vector {
private: 
    T* data;
    unsigned int length;

    template <typename U>
    void sizeCheck(const Vector<U>& _vector) const {
        if(length == 0 || length != _vector.len()) {
            throw std::logic_error("length does not fit."); //exception 
        }
    }

public:
    // default constructor
    Vector() : data(nullptr), length(0) {}

    // copy constructor
    Vector(const Vector<T>& _vector) : length(_vector.length) {
        data = new T[_vector.length];
        memcpy(data, _vector.data, _vector.length * sizeof(T));
    }

    // move constructor
    Vector(Vector&& _vector) : data(_vector.data) , length(_vector.length) {
        _vector.data = nullptr;
        _vector.length = 0;
    }

    // length constructor
    Vector(unsigned int _length) : length(_length) {
        data = new T[_length];
        memset(data, 0, _length * sizeof(T));
    }

    // intializer list constructor
    Vector(std::initializer_list<T> l) : length(l.size()) {
        data = new T[l.size()];
        std::copy(l.begin(), l.end(), data);
    }

    // destructor
    ~Vector() {
        delete[] data;
    }
    
    // copy assignment operator
    Vector<T>& operator = (const Vector<T>& _vector) {
        if (this != &_vector) {
            // Free the existing resource.
            delete[] data;

            length = _vector.length;
            data = new T[_vector.len()];

            std::copy(_vector.data, _vector.data + _vector.len(), data);
        }

        return *this;
    }

    // move assignment operator
    Vector<T>& operator = (Vector<T>&& _vector) {
        if (this != &_vector) {
            // Free the existing resource.
            delete[] data;

            data = _vector.data;
            length = _vector.len();

            _vector.data = nullptr;
            _vector.length = 0;
        }

        return *this;
    }

    // [] operator
    T& operator [] (unsigned int index) {
        return *(data + index);
    }

    const T& operator [] (unsigned int index) const {
        return *(data + index);
    }

    // + operator
    template <typename U>
    auto operator + (const Vector<U>& _vector) const {
        sizeCheck(_vector);

        Vector<typename std::common_type<T, U>::type> resVec(length);

        // FIXME : loop unrolling maybe
        for(unsigned int i = 0; i < length; ++i) {
            resVec[i] = data[i] + _vector[i];
        }

        return resVec;
    }

    // - operator
    template <typename U>
    auto operator - (const Vector<U>& _vector) const {
        sizeCheck(_vector);

        Vector<typename std::common_type<T, U>::type> resVec(length);

        for(unsigned int i = 0; i < length; ++i) {
            resVec[i] = data[i] - _vector[i];
        }

        return resVec;
    }

    // scalar multiplication operator
    template <typename U>
    auto operator * (const U& _scalar) const {
        Vector<typename std::common_type<T, U>::type> resVec(length);

        for(unsigned int i = 0; i < length; ++i) {
            resVec[i] = _scalar * data[i];
        }

        return resVec;
    }

// iterator realization
private:
    class vecIter : std::iterator<std::input_iterator_tag, T> {
    private:
        T* p;
    public:
        vecIter(T* x) : p(x) {}
        vecIter(const vecIter& mit) : p(mit.p) {}

        vecIter& operator ++ () {
            ++p; 
            return *this;
        }

        vecIter operator ++ (int) {
            vecIter tmp(*this); 
            operator ++ (); 

            return tmp;
        }

        bool operator == (const vecIter& rhs) const { return p == rhs.p; }
        bool operator != (const vecIter& rhs) const { return p != rhs.p; }
        T& operator *() { return *p; }
    };

public:
    auto begin() const {
        return vecIter(data);
    }

    auto end() const {
        return vecIter(data + length);
    }

    // utilities
public:
    // length function
    const unsigned int len() const {
        return length;
    }

    void print() {
        for (auto iter = this->begin(); iter != this->end(); ++iter)
            std::cout << *iter << std::endl;
    }

    bool empty() const {
        return (length == 0);
    }

    void reset() {
        memset(data, 0, length * sizeof(T));
    }
};

// ------------------------------------------------------------ // 
// Vector functions

template<typename T, typename U>
typename std::common_type<T, U>::type 
dot(const Vector<T>& lhs, 
    const Vector<U>& rhs)
{
    if(lhs.len() == 0 || rhs.len() == 0 || lhs.len() != rhs.len()) {
        throw std::logic_error("length does not fit."); //exception 
    }

    // typename std::common_type<T, U>::type sum(0);
    
    /*
    for(unsigned int i = 0; i < lhs.len(); ++i) {
        sum += lhs[i] * rhs[i];
    }

    return sum;
    */

    return std::inner_product(lhs.begin(), lhs.end(), rhs.begin(), 
                static_cast<typename std::common_type<T, U>::type>(0));
}

// scalar multiplication
template <typename T, typename U>
auto operator* (const U& _scalar, const Vector<T>& _vector) {
    return _vector * _scalar;
}

// all-one vector generator
template<typename T>
Vector<T> allOneVecGen(unsigned int _size) {
    Vector<T> allOneVec(_size);

    for (auto iter = allOneVec.begin(); iter != allOneVec.end(); ++iter) {
        *iter = 1;
    }

    /*
    for (unsigned int i = 0; i < _size; ++i) {
        allOneVec[i] = 1;
    }
    */

    return allOneVec;
}


// ------------------------------------------------------------ //
// class Matrix 

template <typename T>
class Matrix {
private:
    std::map<std::pair<unsigned int, unsigned int>, T> mat;
    const unsigned int rowNum;
    const unsigned int colNum;

    // boundary check
    void boundaryCheck(const std::pair<unsigned int, unsigned int>& pos) const {
        if (pos.first >= rowNum || pos.second >= colNum)
            throw std::logic_error("out of matrix range");
    }

public:
    // size constructor
    Matrix (unsigned int _rowNum, unsigned int _colNum) : rowNum(_rowNum), colNum(_colNum) {}

    /*
    // copy constructor
    Matrix (const Matrix<T>& _matrix) : rowNum(_matrix.rowNum), colNum(_matrix.colNum) {
        std::copy(_matrix.mat.begin(), _matrix.mat.end(), mat.begin());
    }
    */

    // destructor
    ~Matrix() {}

    // operator[](const std::pair<int, int>& ij)
    T& operator [] (const std::pair<unsigned int, unsigned int>& pos) {
        boundaryCheck(pos);

        return mat[pos];
    }
    
    // operator()(const std::pair<int, int>& ij) const
    const T& operator () (const std::pair<unsigned int, unsigned int>& pos) {
        boundaryCheck(pos);

        if (!mat.count(pos))
            throw std::logic_error("such element is not present");
        
        return mat[pos];
    }


    // copy assignment
    Matrix<T>& operator = (const Matrix<T>& _matrix) {
        if (this != &_matrix) {
            // Free the existing resource.
            mat.clear();

            rowNum = _matrix.rowNum;
            colNum = _matrix.colNum;

            std::copy(_matrix.mat.begin(), _matrix.mat.end(), mat.begin());
        }

        return *this;
    }

    /*
    // scalar multiplication
    template<typename U>
    Matrix<typename std::common_type<T, U>::type>
    operator * (const U& _scalar) {
        Matrix<typename std::common_type<T, U>::type> resMat(*this);

        for (auto iter = resMat.begin(); iter != resMat.end(); ++iter) {
            iter->second *= _scalar;
        }

        return resMat;
    }
    */

    /*
    template<typename U>
    Vector<typename std::common_type<T, U>::type>
    operator * (const Vector<U>& _vector) { 
        if (colNum != _vector.len())
            throw std::logic_error("size not compatible");

        Vector<typename std::common_type<T, U>::type> resVec(rowNum);

        for (auto iter = mat.begin(); iter != mat.end(); ++iter) {
            resVec[iter->first.first] += iter->second * _vector[iter->first.second];
        }

        return resVec;
    }
    */

    // iterator realization
public:
    auto begin() const {
        return mat.begin();
    }

    auto end() const {
        return mat.end();
    }

    // utilities
public:
    const std::pair<unsigned int, unsigned int> size() const {
        return std::pair<unsigned int, unsigned int>(rowNum, colNum);
    }

    bool empty() const {
        return mat.empty();
    }

    void print() {
        for(auto iter = mat.begin(); iter != mat.end(); ++iter) {
            std::cout << iter->first.first << ", " << iter->first.second << "->" << iter->second << std::endl;
        }
    }
};

// ------------------------------------------------------------ // 
// Matrix functions

// matrix multiplication
template<typename T, typename U>
Vector<typename std::common_type<T, U>::type>
operator * (const Matrix<T>& lhs, const Vector<U>& rhs) {
    // if(lhs.empty() || rhs.empty())
    //    throw std::logic_error("empty candidate(s)");

    auto matSize = lhs.size();
    if (matSize.second != rhs.len())
        throw std::logic_error("size not compatible");

    Vector<typename std::common_type<T, U>::type> resVec(matSize.first);

    for (auto iter = lhs.begin(); iter != lhs.end(); ++iter) {
        resVec[iter->first.first] += iter->second * rhs[iter->first.second];
    }

    return resVec;
}

/*
// scalar multiplication
template<typename T, typename U>
Matrix<typename std::common_type<T, U>::type>
operator * (const U& _scalar, const Matrix<T>& _matrix) {
    return _matrix * _scalar;
}
*/

// identity matrix generator
template<typename T>
Matrix<T> unitMatGen(unsigned int _size) {
    Matrix<T> unitMat(_size, _size);

    for (unsigned int i = 0; i < _size; ++i) {
        unitMat[{i, i}] = 1;
    }

    return unitMat;
}

// ------------------------------------------------------------ // 
// function cg

template<typename T>
int cg(const Matrix<T>& A, 
       const Vector<T>& b, 
       Vector<T>&       x, 
       T                tol     =(T)1e-8, 
       int              maxiter = 100)
{
    /*
    if (A.empty() || b.empty()) 
        throw std::logic_error("empty candidate(s)");
    */

    const T total_square(tol * tol);
    
    Vector<T> p(b - A * x);
    Vector<T> r(p);
    // Vector<T> r_cur(p), r_next(r_cur); 
    T alpha_k(0), beta_k(0);
    
    for (int k = 0; k < maxiter; ++k) {
        const T r_cur_innerProduct(dot(r, r));
        
        alpha_k = r_cur_innerProduct / dot((A * p), p);

        x = x + alpha_k * p ;

        r = r -  A * p * alpha_k; // r_next

        const T r_next_innerProduct(dot(r, r));
        if (r_next_innerProduct < total_square) { return k; }

        beta_k = r_next_innerProduct / r_cur_innerProduct;
        p = r +  beta_k * p;
    }
    
    return -1;
}

// ------------------------------------------------------------ // 
// class Heat

template <int n, typename T> // n = dimension
class Heat {
private:
    const T alpha;
    const unsigned int point_per_dim;
    const unsigned int total_point;
    const T dt;
    const T dx;

    Matrix<T> M;
    Vector<T> u_0_vec;

    bool isRightBorder(unsigned int idx) {
        return ((idx + 1) % point_per_dim == 0);
    }

public:
    Heat(T _alpha, int _point_per_dim, T _dt) 
        : alpha(_alpha), point_per_dim(_point_per_dim), 
        total_point(std::pow(_point_per_dim, n)),
        dt(_dt), dx(1 / double(_point_per_dim + 1)), 
        M(total_point, total_point), u_0_vec(total_point)
    {
        const T dx_square = std::pow(dx, 2);
        const double M_coeff(_alpha * (dt / dx_square));

        // const unsigned int iterNum((total_point + 1) >> 1);
        const unsigned int iterNum(total_point);
        for (unsigned int i = 0; i < iterNum; ++i) {
            M[{i, i}] = 1 + M_coeff * 2 * n;

            unsigned int neighborDistance(1);
            unsigned int divisor(total_point / point_per_dim);
            T u_0(1);

            for (unsigned int dim = 0; dim < n; ++dim) {
                const unsigned int neighborPos(i + neighborDistance);

                if ((neighborPos < total_point) && !(isRightBorder(i) && neighborDistance == 1)) {
                    M[{i, neighborPos}] = M[{neighborPos, i}] 
                                                    = -M_coeff;
                }

                neighborDistance *= point_per_dim;

                // positioning
                u_0 *= sin(M_PI * ((i / divisor) % point_per_dim + 1) * dx);

                divisor /= point_per_dim;
            }

            if (u_0 > 1e-10) { 
                // u_0_vec[i] = u_0_vec[total_point - i - 1] = u_0; 
                u_0_vec[i] = u_0;
            }
        }

        std::cout << "u_0_vec" << std::endl;
        u_0_vec.print();
    }
    
    Vector<T> exact(T t) const {
        return (u_0_vec * (exp(-n * M_PI * M_PI * alpha * t)));
    }

    Vector<T> solve(T t) const {
        Vector<T> resVec(total_point);
		Vector<T> init(u_0_vec);

        const unsigned int iterNum((unsigned int)(t / dt));
		for(unsigned int i = 0; i < iterNum; ++i) {
            resVec.reset();
            
			// std::cout << "success: " << cg<T>(M, init, resVec, 1e-10, 1000) << std::endl;
			cg<double>(M, init, resVec, 1e-8, 1000);

            init = resVec;

            // resVec.print();
		}

		return resVec;
    }

    // utilities
    void printMatrix() {
        M.print();
    }
};

// ------------------------------------------------------------ // 

int main(int argc, char* argv[]) {
    Vector<float> vec_f(3);
    Vector<int> vec_i(3);

    auto vec = vec_f + vec_i;

    std::cout << typeid(vec[0]).name() << std::endl;

    std::cout << dot(vec_f, vec_i) << std::endl;

    std::initializer_list<float> lf{1.0f, 2.0f, 3.0f};
    std::initializer_list<int> li{1, 2, 3};

    Vector<float> vec_f_2(lf);
    Vector<int> vec_i_2(li);

    std::cout << std::inner_product(vec_f_2.begin(), vec_f_2.end(), vec_f_2.begin(), 0) << std::endl;

    auto res = dot(vec_f_2, vec_i_2);

    std::cout << res << std::endl;

    std::cout << typeid(res).name() << std::endl;

    Matrix<float> mat(3, 3);
    mat[{0, 1}] = 2.0;

    auto resVec = mat * vec_f_2;

    resVec = allOneVecGen<float>(2);
    resVec.print();

    auto unitMat = unitMatGen<float>(27);
    // std::cout << unitMat({1, 1}) << std::endl;

    // std::cout << cg<float>(mat, vec_f_2, vec_f) << std::endl;
    // std::cout << cg<float>(unitMat, vec_f, vec_f_2) << std::endl;

    std::cout << "heat here" << std::endl;

    Heat<3, double> heat(0.3125, 3, 1e-3);
    auto r = heat.exact(0.5);
    std::cout << "exact result starts here." << std::endl;
    r.print();

    auto s = heat.solve(0.5);
    std::cout << "solve result starts here." << std::endl;
    s.print();

    auto diff(r - s);
    std::cout << "diff result starts here." << std::endl;
    diff.print();

    heat.printMatrix();

    return 0;
}

/*
0.000535323
0.000776638
0.000577703
0.000866067
0.00126053
0.000975188
0.000808525
0.00115728
0.00104717
*/