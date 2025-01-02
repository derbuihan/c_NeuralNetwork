# C Neural Network

Neural Network from scratch in C.

## Create dataset

```bash
conda create -n dataset python=3.12
conda activate dataset
pip install 'scikit-learn' pandas tensorflow
```

```bash
cd dataset
python digits.py
python mnist.py
```

## Build

To build the project, you need to have `gcc` installed. You can build the project using the following commands:

```bash
cd /Users/derbuihan/Programs/C/c_NeuralNetwork
mkdir build
cd build
cmake ..
make
```

## License

This project is licensed under the MIT License.
