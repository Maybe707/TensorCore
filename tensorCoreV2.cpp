#include <initializer_list>
#include <iostream>
#include <iterator>
#include <pthread.h>
#include <cassert>
#include <vector>

template<class T, int row, int col>
class Tensor
{
public:
    int tensorRow;
    int tensorCol;
    T tensor[row][col];
    Tensor() : tensorRow(row), tensorCol(col)
    {
    }

    Tensor(std::initializer_list<std::initializer_list<T>> list)
    {
        *this = list;
    }

    Tensor<T, col, row> tensorTranspose()
        {
            Tensor<T, col, row> tempTranceponentTensor;
            for(int i = 0; i < row; ++i)
                for(int j = 0; j < col; ++j)
                {
                    tempTranceponentTensor[j][i] = tensor[i][j];
                }
            return tempTranceponentTensor;
        }

    template<class T2, int row2, int col2>
    Tensor<T, row, col2> operator*(Tensor<T2, row2, col2>& tensor2);

    T operator()(const int index1, const int index2);
    T* operator[](const int index);
    Tensor<T, row, col> operator-(Tensor<T, row, col>& tensor2);
    Tensor<T, row, col> operator+(Tensor<T, row, col>& tensor2);
    Tensor<T, row, col> operator*(T variable);
    friend Tensor<T, row, col> operator*(T variable, Tensor<T, row, col>& tensor2);
    void operator=(std::initializer_list<std::initializer_list<T>> list);
};

template<class T, int row, int col>
template<typename T2, int row2, int col2>
Tensor<T, row, col2> Tensor<T, row, col>::operator*(Tensor<T2, row2, col2>& tensor2)
{
    assert(col == row2 && "Col of first tensor dont equal row of second tensor");
    Tensor<T, row, col2> tempTensor;
  for (int x = 0; x < row; ++x)
    for (int z = 0; z < col2; ++z) {
        tempTensor.tensor[x][z] = 0;
    }
    for(int i = 0; i < row; ++i)
        for(int j = 0; j < col2; ++j)
            for(int l = 0; l < col; ++l)
            {
                tempTensor.tensor[i][j] += tensor[i][l] * tensor2.tensor[l][j];
            }
    return tempTensor;
}

template <class T, int row, int col>
T Tensor<T, row, col>::operator()(const int index1, const int index2)
{
    return tensor[index1][index2];
}

template <class T, int row, int col>
T* Tensor<T, row, col>::operator[](const int index)
{
    return tensor[index];
}

template<class T, int row, int col>
Tensor<T, row, col> Tensor<T, row, col>::operator-(Tensor<T, row, col>& tensor2)
{
    //assert(tensor2.tensor == col && tensor2.tensorRow == row && "Sizes of tensor dont equal");
    Tensor<T, row, col> tempTensor;
    for(int i = 0; i < row; ++i)
        for(int j = 0; j < col; ++j)
            {
                tempTensor[i][j] = tensor[i][j] - tensor2.tensor[i][j];
            }
    return tempTensor;
}

template<class T, int row, int col>
Tensor<T, row, col> Tensor<T, row, col>::operator+(Tensor<T, row, col>& tensor2)
{
    //assert(tensor2.tensor == col && tensor2.tensorRow == row && "Sizes of tensor dont equal");
    Tensor<T, row, col> tempTensor;
    for(int i = 0; i < row; ++i)
        for(int j = 0; j < col; ++j)
            {
                tempTensor[i][j] = tensor[i][j] + tensor2.tensor[i][j];
            }
    return tempTensor;
}

template<class T, int row, int col>
Tensor<T, row, col> Tensor<T, row, col>::operator*(T variable)
{
    Tensor<T, row, col> tempTensor;
    for(int i = 0; i < row; ++i)
        for(int j = 0; j < col; ++j)
            {
                tempTensor[i][j] = tensor[i][j] * variable;
            }
    return tempTensor;
}

template<class T, int row, int col>
Tensor<T, row, col> operator*(T variable, Tensor<T, row, col>& tensor2)
{
    Tensor<T, row, col> tempTensor;
    for(int i = 0; i < row; ++i)
        for(int j = 0; j < col; ++j)
            {
                tempTensor[i][j] = tensor2[i][j] * variable;
            }

    return tempTensor;
}

template <class T, int row, int col>
std::ostream& operator<<(std::ostream &ostream,
                               Tensor<T, row, col>& tensor)
{
    for(int i = 0; i < row; ++i)
        for(int j = 0; j < col; ++j)
        {
            ostream << tensor.tensor[i][j] << " ";
        }

    return ostream;
}

template <typename X, int row, int col>
Tensor<X, 1, col> cutFunction(Tensor<X, row, col>& tensor, int iteration)
{
    Tensor<X, 1, col> tempTensor;
    for(int i = 0; i < col; ++i)
    {
        tempTensor[0][i] = tensor.tensor[iteration][i];
    }
    return tempTensor;
}

template <class T, int row, int col>
void Tensor<T, row, col>::operator=(std::initializer_list<std::initializer_list<T>> list)
{
    typename std::initializer_list<std::initializer_list<T>>::const_iterator iterator;
    typename std::initializer_list<T>::const_iterator iterator2;
    iterator = list.begin();
    iterator2 = iterator->begin();
    for(int i = 0; i < row; ++i)
    {
        iterator2 = iterator->begin();
        ++iterator;
        for(int j = 0; j < col; ++j)
        {
            tensor[i][j] = *iterator2;
            ++iterator2;
        }
    }
}

// Return X if X > 0; else return 0
template <class T, int row, int col>
void relu(Tensor<T, row, col>& tensor)
{
    for(int i = 0; i < row; ++i)
        for(int j = 0; j < col; ++j)
        {
            if(tensor[i][j] > 0)
                tensor[i][j] = tensor[i][j];
            else
                tensor[i][j] = 0;
        }
        //return (x > 0) * x;
}

// Return 1, if output > 0; else return 0
template<class T, int row, int col>
void relu2deriv(Tensor<T, row, col>& tensor, double* reluResult)
{

    for(int i = 0; i < row; ++i)
        for(int j = 0; j < col; ++j)
        {
            if(tensor[i][j] > 0)
                reluResult[j] = 1;
            else
                reluResult[j] = 0;
        }
}

double getRandomNumber(double min, double max)
{
    static const double fraction = 1.0 / (static_cast<double>(RAND_MAX) + 1.0);
    // Равномерно распределяем рандомное число в нашем диапазоне
    return static_cast<double>(rand() * fraction * (max - min + 1) + min)-1;
}

template <class T, int row, int col>
void weightsGenerator(Tensor<T, row, col>& tensor)
{
    for(int i = 0; i < row; ++i)
    {
        for(int j = 0; j < col; ++j)
        {
            tensor[i][j] = getRandomNumber(0.0, 1.0);
        }
    }
}



class TensorDouble
{
    std::string TensorIdentificator;
    unsigned int Row;
    unsigned int Col;
    double** tensor = new double*[Row];
public:
    TensorDouble(const unsigned int row, const unsigned int col, std::string tenIdent = "unknown") : Row(row), Col(col), TensorIdentificator(tenIdent)
    {
        for(int i = 0; i < row; ++i)
            tensor[i] = new double[col];
    }

    ~TensorDouble()
    {
        for(int i = 0; i < Row; ++i)
            delete [] tensor[i];
        delete [] tensor;
    }

    void setTensorIdentificator(std::string TensIdent) { TensorIdentificator = TensIdent; }
    std::string getTensorIdentificator() { return TensorIdentificator; }

    TensorDouble(std::initializer_list<std::initializer_list<double>> list) : TensorDouble(list.size(), list.begin()->size())
    {
        //this->operator=(list);
        *this = list;
    }

    void SelfTensorTranspose()
    {
        double tempArray[Col][Row];
        for(int p = 0; p < Col; ++p)
            for(int u = 0; u < Row; ++u)
            {
                tempArray[p][u] = tensor[u][p];
            }
        for(int i = 0; i < Row; ++i)
            delete [] tensor[i];
        delete [] tensor;
        unsigned int tempVar = 0;
        tempVar = Row;
        Row = Col;
        Col = tempVar;
        tensor = new double*[Row];
        for(int i = 0; i < Row; ++i)
            tensor[i] = new double[Col];
        for(int j = 0; j < Row; ++j)
            for(int z = 0; z < Col; ++z)
            {
                this->tensor[j][z] = tempArray[j][z];
            }
    }

    TensorDouble tensorTransposeMethod()
    {
        TensorDouble TensorTranspose(Col, Row);
        TensorDouble tempTensor(Col, Row);
        for(int i = 0; i < Row; ++i)
            for(int j = 0; j < Col; ++j)
            {
                TensorTranspose[j][i] = tensor[i][j];
            }
        return  TensorTranspose;
    }

    void operator=(const TensorDouble& tensor2);
    TensorDouble operator*(const TensorDouble& tensor2);

    double operator()(const int index1, const int index2);
    double* operator[](const int index);
    const double* operator[](const int index) const;
    void operator-(TensorDouble& tensor2);
    void operator+(TensorDouble& tensor2);
    void operator*(double variable);
    friend void operator*(double variable, TensorDouble& tensor2);
    friend std::ostream& operator<<(std::ostream &ostream, TensorDouble& tensor);
    friend TensorDouble cutFunction(TensorDouble& tensor, int iteration);
    void operator=(std::initializer_list<std::initializer_list<double>> list);
    friend void reluDoubleSpec(TensorDouble& tensor);
    friend TensorDouble& relu2derivDoubleSpec(TensorDouble& tensor, TensorDouble& reluResult);
    friend void weightsGeneratorDoubleSpec(TensorDouble& tensor);
};

void TensorDouble::operator=(const TensorDouble& tensor2)
{
    if((this->Row == tensor2.Row) || (this->Col == tensor2.Col))
    {
        for(int i = 0; i < Row; ++i)
                delete [] tensor[i];
            delete [] tensor;
        this->Row = tensor2.Row;
        this->Col = tensor2.Col;
        tensor = new double*[Row];
        for(int i = 0; i < Row; ++i)
            tensor[i] = new double[Col];
    }
    for(int j = 0; j < Row; ++j)
        for(int z = 0; z < Col; ++z)
        {
            this->tensor[j][z] = tensor2[j][z];
        }
}

TensorDouble TensorDouble::operator*(const TensorDouble& tensor2)
{
    assert(Col == tensor2.Row && "Col of first tensor dont equal row of second tensor");
    TensorDouble TensorMultiply(Row, tensor2.Col);
  for (int x = 0; x < Row; ++x)
    for (int z = 0; z < tensor2.Col; ++z) {
        TensorMultiply[x][z] = 0;
    }
    for(int i = 0; i < Row; ++i)
        for(int j = 0; j < tensor2.Col; ++j)
            for(int l = 0; l < Col; ++l)
            {
                TensorMultiply[i][j] += tensor[i][l] * tensor2.tensor[l][j];
            }
    return TensorMultiply;
}

double TensorDouble::operator()(const int index1, const int index2)
{
    return tensor[index1][index2];
}

double* TensorDouble::operator[](const int index)
{
    return tensor[index];
}

const double* TensorDouble::operator[](const int index) const
{
    return tensor[index];
}

void TensorDouble::operator-(TensorDouble& tensor2)
{
    assert(tensor2.Col == Col && tensor2.Row == Row && "Sizes of tensor dont equal");
    int tempTensor[Row][Col];
    for(int i = 0; i < Row; ++i)
        for(int j = 0; j < Col; ++j)
            {
                tempTensor[i][j] = tensor[i][j] - tensor2.tensor[i][j];
            }
    for(int z = 0; z < Row; ++z)
        for(int x = 0; x < Col; ++x)
            {
                tensor[z][x] = tempTensor[z][x];
            }
}

void TensorDouble::operator+(TensorDouble& tensor2)
{
    assert(tensor2.Col = Col && tensor2.Row == Row && "Sizes of tensor dont equal");
    int tempTensor[Row][Col];
    for(int i = 0; i < Row; ++i)
        for(int j = 0; j < Col; ++j)
            {
                tempTensor[i][j] = tensor[i][j] + tensor2.tensor[i][j];
            }
    for(int z = 0; z < Row; ++z)
        for(int x = 0; x < Col; ++x)
            {
                tensor[z][x] = tempTensor[z][x];
            }
}

void TensorDouble::operator*(double variable)
{
    int tempTensor[Row][Col];
    for(int i = 0; i < Row; ++i)
        for(int j = 0; j < Col; ++j)
            {
                tempTensor[i][j] = tensor[i][j] * variable;
            }
    for(int z = 0; z < Row; ++z)
        for(int x = 0; x < Col; ++x)
            {
                tensor[z][x] = tempTensor[z][x];
            }
}

void operator*(double variable, TensorDouble& tensor2)
{
    int tempTensor[tensor2.Row][tensor2.Col];
    for(int i = 0; i < tensor2.Row; ++i)
        for(int j = 0; j < tensor2.Col; ++j)
            {
                tempTensor[i][j] = tensor2.tensor[i][j] * variable;
            }
    for(int z = 0; z < tensor2.Row; ++z)
        for(int x = 0; x < tensor2.Col; ++x)
            {
                tensor2.tensor[z][x] = tempTensor[z][x];
            }
}

std::ostream& operator<<(std::ostream &ostream, TensorDouble& tensor)
{
    std::cout << std::endl;
    std::cout << tensor.getTensorIdentificator() << std::endl;
    for(int i = 0; i < tensor.Row; ++i)
    {
        std::cout << std::endl;
        for(int j = 0; j < tensor.Col; ++j)
        {
            ostream << tensor.tensor[i][j] << " ";
        }
    }
    return ostream;
}

TensorDouble cutFunction(TensorDouble& tensor, int iteration)
{
    TensorDouble TensorCut(1, tensor.Col);
    for(int i = 0; i < tensor.Col; ++i)
    {
        TensorCut[0][i] = tensor.tensor[iteration][i];
    }
    return TensorCut;
}

void TensorDouble::operator=(std::initializer_list<std::initializer_list<double>> list)
{

    std::initializer_list<std::initializer_list<double>>::const_iterator iterator;
    std::initializer_list<double>::const_iterator iterator2;
    iterator = list.begin();
    iterator2 = iterator->begin();
    for(int i = 0; i < Row; ++i)
    {
        iterator2 = iterator->begin();
        ++iterator;
        for(int j = 0; j < Col; ++j)
        {
            tensor[i][j] = *iterator2;
            ++iterator2;
        }
    }
    //return *this;
}

// Return X if X > 0; else return 0
void reluDoubleSpec(TensorDouble& tensor)
{
    for(int i = 0; i < tensor.Row; ++i)
        for(int j = 0; j < tensor.Col; ++j)
        {
            if(tensor[i][j] > 0)
                tensor[i][j] = tensor[i][j];
            else
                tensor[i][j] = 0;
        }
        //return (x > 0) * x;
}

// Return 1, if output > 0; else return 0
TensorDouble& relu2derivDoubleSpec(TensorDouble& tensor, TensorDouble& reluResult)
{

    for(int i = 0; i < tensor.Row; ++i)
        for(int j = 0; j < tensor.Col; ++j)
        {
            if(tensor[i][j] > 0)
                reluResult[i][j] = 1;
            else
                reluResult[i][j] = 0;
        }
    return reluResult;
}

void weightsGeneratorDoubleSpec(TensorDouble& tensor)
{
    for(int i = 0; i < tensor.Row; ++i)
    {
        for(int j = 0; j < tensor.Col; ++j)
        {
            tensor[i][j] = getRandomNumber(0.0, 1.0);
        }
    }
}
