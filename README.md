English description below

Classificador MLP com Keras Tuner

Descrição do Projeto

Este projeto explora a utilização de redes neurais Multilayer Perceptron (MLP) para classificação de imagens do dataset Fashion MNIST. 

Utilizei o Keras Tuner para otimizar hiperparâmetros, como o número de camadas ocultas, número de neurônios por camada, taxa de aprendizado e otimizadores. 

O objetivo foi maximizar a acurácia de validação por meio de um processo de busca sistemática.

Passos:

Pré-processamento dos dados: 

Normalização dos dados de entrada para garantir que os valores fiquem entre 0 e 1.

Modelo customizado: 

Desenvolvi uma função para criar o modelo MLP de forma dinâmica, com a possibilidade de ajustar os hiperparâmetros.

Tuning de Hiperparâmetros: 

Usei o Keras Tuner (Hyperband) para explorar diferentes configurações de hiperparâmetros.

Callbacks: 

Adicionei callbacks como EarlyStopping e TensorBoard para monitorar e otimizar o treinamento do modelo.

Treinamento e Avaliação: 

Treinei o modelo nos dados de treino e validei o desempenho nos dados de teste.

Predição Visual:

Gerei previsões em imagens novas e exibi os resultados graficamente.

Estrutura do Projeto

Otimização de hiperparâmetros: 

A busca pelos melhores hiperparâmetros foi realizada utilizando o algoritmo Hyperband, que explorou configurações de rede com até 8 camadas ocultas e dois otimizadores (SGD e Adam).

Arquitetura do Modelo: 

A rede neural criada possui uma camada de entrada que achata (flatten) os dados, um número dinâmico de camadas densas com ativação ReLU, e uma camada final com softmax para classificação em 10 classes.

Resultados do Tuning: O modelo alcançou uma acurácia de validação de 90,1% com os seguintes hiperparâmetros:

5 camadas ocultas

224 neurônios por camada

Taxa de aprendizado: 0.00046

Otimizador: Adam

Resultados Obtidos

Acurácia no Teste: 88,7%.

Previsões Exemplares: A rede neural foi capaz de identificar corretamente itens de vestuário, como botas, blusas e calças, em novas imagens.

Visualização de Previsões: As predições foram exibidas junto com as imagens correspondentes, permitindo verificar a precisão do modelo.

Ferramentas Utilizadas

Python: 

Linguagem principal do projeto.

TensorFlow/Keras: 

Frameworks para criação e treinamento do modelo.

Keras Tuner:

Biblioteca utilizada para otimização de hiperparâmetros.








MLP Classifier with Keras Tuner

Project Description

This project explores the use of Multilayer Perceptron (MLP) neural networks for image classification using the Fashion MNIST dataset.

Keras Tuner was employed to optimize hyperparameters, such as the number of hidden layers, the number of neurons per layer, the learning rate, and optimizers.

The goal was to maximize validation accuracy through a systematic search process.

Steps

Data Preprocessing:

I normalized input data to ensure values are within the range of 0 to 1.

Custom Model:

I Developed a function to dynamically create the MLP model, allowing for hyperparameter adjustments.

Hyperparameter Tuning:

i Used Keras Tuner (Hyperband) to explore various hyperparameter configurations.
Callbacks:

i Added callbacks such as EarlyStopping and TensorBoard to monitor and optimize the training process.

Training and Evaluation:

i Trained the model on the training data and validated performance on test data.

Visual Prediction:

i generated predictions for new images and displayed the results graphically.

Project Structure

Hyperparameter Optimization:

Hyperband algorithm was used to search for the best hyperparameters, exploring network configurations with up to 8 hidden layers and two optimizers (SGD and Adam).

Model Architecture:

The neural network includes:

An input layer that flattens the data.

A dynamic number of dense layers with ReLU activation.

A final layer with softmax activation for classification into 10 classes.

Tuning Results:

The model achieved a validation accuracy of 90.1% with the following hyperparameters:

5 hidden layers

224 neurons per layer

Learning rate: 0.00046

Optimizer: Adam

Results Obtained

Test Accuracy: 88.7%

Exemplary Predictions:

The neural network successfully identified clothing items such as boots, shirts, and trousers in new images.

Prediction Visualization:

Predictions were displayed alongside the corresponding images, allowing for validation of model accuracy.

Tools Used

Python:

The primary language for this project.

TensorFlow/Keras:

Frameworks for building and training the model.

Keras Tuner:

Library used for hyperparameter optimization.

Matplotlib:

For data and result visualization.

Pandas and NumPy:

For data manipulation.

Matplotlib: 

Para visualização de dados e resultados.

Pandas e NumPy: 

Para manipulação de dados.


