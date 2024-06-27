[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Multi-Objective Linear Ensembles for Robust and Sparse Training of Few-Bit Neural Networks

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).


## Cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2023.0281

https://doi.org/10.1287/ijoc.2023.0281.cd

Below is the BibTex for citing this snapshot of the repository.

```
@misc{INN-mip-training,
  author =        {Bernardelli, Ambrogio Maria and Gualandi, Stefano and Milanesi, Simone and Lau, Hoong Chuin and Yorke-Smith, Neil},
  publisher =     {INFORMS Journal on Computing},
  title =         {{Multi-Objective Linear Ensembles for Robust and Sparse Training of Few-Bit Neural Networks}},
  year =          {2024},
  doi =           {10.1287/ijoc.2023.0281.cd},
  url =           {https://github.com/INFORMSJoC/2023.0281},
  note =          {Available for download at https://github.com/INFORMSJoC/2023.0281},
}  
```

## Description

The goal of this script is to train neural networks with integer weights with MILP models and to test their accuracy.

## Usage

There is only one python script that can be executed.

The packages needed for the code are the following: `gurobipy`, `time`, `keras.dataset`, `numpy`. 
The PARAMETERS (starting from line 382) can be modified accordingly to the experiments.

For every value in the list `different_p`, the file produces 3*nm* + 1 csv files
where *n* is the number of instances, namely `instances`, and *m* is the number of different numbers of
training images, namely the length of the list `training_images`
Fixed the training images *i* and the instance *j*, the file produces
- `labels_i_j.csv`
    for every image used in the tests, how many networks labelled a certain image with a certain label
- `test_inn_i_j.csv`
    for every network of the ensemble, some infos about gap, time, etc., are collected
- `test_inn_i_j_weights.csv`
    for every network of the ensemble, the weights distribution is collected.
  
The last file, `test_inn.csv`, contains results on accuracy and label statuses. All the results are obtained by analizing the generated csv files.

## Results

Figure 1 compares
* the test accuracy of the four hierarchical models (left);
* the percentages of non-zero weights of the three trained models (right).
The data tested are images of 4s and 9s of the MNIST dataset.

![Figure 1](results/Figure-1.pdf)

Figure 2 compares the results of our BeMi ensemble with the four other methods of the literature on the whole MNIST dataset.

![Figure 2](results/Figure-2.pdf)

Figure 3 compares
* the test accuracy on the whole MNIST dataset (left);
* the test accuracy on the whole Fashion-MNIST dataset (right).

![Figure 3](results/Figure-3.pdf)

Figure 4 compares the accuracy of different INNs by varying the set of the weights. The data tested are images of 4s and 9s of the MNIST dataset.

![Figure 4](results/Figure-4.pdf)

Figure 5 depicts a confusion matrix for the accuracy of BNNs trained on 40 images of the MNIST dataset. 

![Figure 5](results/Figure-5.pdf)

Figure 6 replicates the results shown in Figure 1 but with images of 1s and 8s of the MNIST dataset.

![Figure 6](results/Figure-6.pdf)

## Replicating

TO replicate the experiments, the PARAMETERS have to be changed accordingly. In addition, to switch from the MNIST to the Fashion-MNIST dataset, one has to change line 409 from 
```
(train_X, train_y), (test_X, test_y) = mnist.load_data()
```
to
```
(train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
```
and line 30 from
```
from keras.datasets import mnist
```
to
```
from keras.datasets import mnist
```


## Data

The data used in the experiments are the MNIST and Fashion-MNIST dataset, imported through the `keras.dataset` package.
