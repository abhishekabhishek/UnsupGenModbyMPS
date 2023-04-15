# Synergy between Quantum Circuits and Tensor Networks
This repository is a software implementation of the research paper : https://arxiv.org/pdf/2208.13673.pdf

The solution comprises of three main parts:
* Training Matrix Product State (MPS) tensor network on bars and stripes dataset
* Mapping MPS tensor network to Parameterized Quantum Circuits (PQC)
* Further extending the PQC using quantum resources

## Deliverables

* Presentation Slides
  * PDF file : https://github.com/abhishekabhishek/UnsupGenModbyMPS/blob/master/Final_Presentation.pdf
  * PPTX file: https://github.com/abhishekabhishek/UnsupGenModbyMPS/blob/master/Final_Presentation.pptx
* Project Report - 
* Statement of Contribution - 
* Software Implementation
  * MPS training - adapted from pre-existing implementation done by authors of the paper:
    > [**Unsupervised Generative Modeling Using Matrix Product States** by *Zhao-Yu Han, Jun Wang, Heng Fan, Lei Wang, Pan Zhang*](https://arxiv.org/abs/1709.01662)

## Testing 
In order to get an understanding of the project, run the following jupyter notebook: sandbox/random_near_identity_extended_circuit.ipynb

## Files added
For this project we added the following files:
 
 * random_near_identity_extended_circuit - This file has methods to create as well as train the fully ranodmly and near unitary initialized circuits, take results from MPS and convert them into a quantum circuit, extend the quantum circuit created from MPS with SU(4) unitaries, and train the extended quantum circuit.
 * 
