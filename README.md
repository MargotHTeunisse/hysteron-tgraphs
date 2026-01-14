#About

The code in this repository was published in conjunction with the article "Transition Graphs of Interacting Hysterons: Structure, Design, Organisation and Statistics" by Margot H. Teunisse and Martin van Hecke for Royal Society Open Science (2025). 

The notebook 'walkthrough.ipynb' in the repository github.com/MargotHTeunisse/hysteron-tgraphs reproduces the results detailed in the article.

#Getting Started

To import this software, first install it from PyPI by running 

```console

pip install hysteron_tgraphs

```
**Dependencies**

The core functionalities of this code are compatible with

Python >= 3.12

Numpy >= 1.26.4

Scipy >= 1.13.1

**Optional features**

These features have optional dependencies which can be specified on installation, e.g. by running 


```console

pip install hysteron_tgraphs[plotting]

```

plotting : Contains functions for visualizing transition graphs and switching fields. Requires Matplotlib >= 3.9.2. 

graph-analysis: Currently only used to remove Garden-of-Eden states. Requires NetworkX >= 3.3.

polyhedron-analysis: <span style="color:red"> Not recommended for most use cases. </span> Calculates the probability of a transition graph by calculating the volume of the corresponding polyhedron. Uses Pycddlib 2.1.8.post1. 


#Contact Information

For questions please contact:

Margot Teunisse

Email: teunisse@physics.leidenuniv.nl

#How to Cite

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16920053.svg)](https://doi.org/10.5281/zenodo.16920053)

#Financial Statement

This work is supported by ERC-101019474.

<img src="https://erc.europa.eu/sites/default/files/2025-08/LOGO_ERC-FLAG_EU-no%20text.png" width="200"/>
