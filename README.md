# The Persistent Laplacian for Data Science

This repo contains the code for the ICML 2023 paper "[The Persistent Laplacian for Data Science: Evaluating Higher-Order Persistent Spectral Representations of Data](https://proceedings.mlr.press/v202/davies23c.html)".

> Persistent homology is arguably the most successful technique in Topological Data Analysis. It combines homology, a topological feature of a data set, with persistence, which tracks the evolution of homology over different scales. The persistent Laplacian is a recent theoretical development that combines persistence with the combinatorial Laplacian, the higher-order extension of the well-known graph Laplacian. Crucially, the Laplacian encodes both the homology of a data set, and some additional geometric information not captured by the homology. Here, we provide the first investigation into the efficacy of the persistent Laplacian as an embedding of data for downstream classification and regression tasks. We extend the persistent Laplacian to cubical complexes so it can be used on images, then evaluate its performance as an embedding method on the MNIST and MoleculeNet datasets, demonstrating that it consistently outperforms persistent homology.

## Requirements

Run `pip install -r requirements.txt` to install requirements.

## Usage 

The functions to compute the persistent Laplacian in a number of ways are in `src/perslap.py`.

- To use the persistent Laplacian as a feature vector for a filtration $K_0 \subset K_1 \subset \dots$ using our vectorisation scheme, use the function `features_perslap`. 
- To compute the persistent Laplacian of a complex pair $K \subset L$, use the function `pers_lap_pair`.
- To compute the non-persistent higher Laplacian of a simplicial or cubical complex $K$, use the function `hodge_laplacian`.

## Citing

If you use this code, please cite our paper.

```
@InProceedings{pmlr-v202-davies23c,
  title = 	 {The Persistent {L}aplacian for Data Science: Evaluating Higher-Order Persistent Spectral Representations of Data},
  author =       {Davies, Thomas and Wan, Zhengchao and Sanchez-Garcia, Ruben J},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {7249--7263},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
}
```
