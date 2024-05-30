# FourierKAN-GCF: Fourier Kolmogorov-Arnold Network - An Effective and Efficient Feature Transformation for Graph Collaborative Filtering

<!-- PROJECT LOGO -->

## Introduction

This is the Pytorch implementation for our FourierKAN-GCF [paper]():

>FourierKAN-GCF: Fourier Kolmogorov-Arnold Network - An Effective and Efficient Feature Transformation for Graph Collaborative Filtering

## Discussion

Rethinking feature transformation component in GCNs in recommendation field!

[LightGCN](https://arxiv.org/pdf/2002.02126) simplifies [NGCF](https://arxiv.org/pdf/1905.08108) by remove feature transformation, formally:

**NGCF**
```math
\begin{aligned} \mathbf{e}_u^{(l+1)} & =\sigma(\mathbf{W}_1 \mathbf{e}_u^{(l)}+\sum_{i \in \mathcal{N}_u} \frac{\mathbf{W}_1 \mathbf{e}_i^{(l)}+\mathbf{W}_2(\mathbf{e}_i^{(l)} \odot \mathbf{e}_u^{(l)})}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}}), \\ \mathbf{e}_i^{(l+1)} & =\sigma(\mathbf{W}_1 \mathbf{e}_i^{(l)}+\sum_{u \in \mathcal{N}_i} \frac{\mathbf{W}_1 \mathbf{e}_u^{(l)}+\mathbf{W}_2(\mathbf{e}_u^{(l)} \odot \mathbf{e}_i^{(l)})}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}}).
\end{aligned}
```
**LightGCN**
```math
\mathbf{e}_u^{(l+1)} =\sum_{i \in \mathcal{N}_u} \frac{\mathbf{e}_i^{(l)}}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}}, \quad \mathbf{e}_i^{(l+1)} =\sum_{u \in \mathcal{N}_i} \frac{\mathbf{e}_u^{(l)}}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}}.
```
We point out that $\mathbf{W}_1$ is unnecessary, but interaction part $\mathbf{W}_2(\mathbf{e}_i^{(l)} \odot \mathbf{e}_u^{(l)}$ is valuable for recommendation task, but it's hard to train on sparsity dataset.

Thanks to the original implementations [KAN](https://github.com/KindXiaoming/pykan) and [FourierKAN](https://github.com/GistNoesis/FourierKAN).

We use single-layer FourierKAN to replace MLP in feature transformation component and achieve better results than LightGCN and NGCF on MOOC and Amazon Games datasets. Formally:

**FourierKAN-GCF**
```math
\phi_F(\mathbf{x})=\sum_{i=1}^{d} \sum_{k=1}^{g}\left(\cos \left(k \mathbf{x}_i\right) \cdot a_{i k}+\sin \left(k \mathbf{x}_i\right) \cdot b_{i k}\right).
```
```math
\begin{aligned} \mathbf{e}_u^{(l+1)} & =\sigma(\mathbf{e}_u^{(l)}+\sum_{i \in \mathcal{N}_u} \frac{\mathbf{e}_i^{(l)}+\phi_F(\mathbf{e}_i^{(l)} \odot \mathbf{e}_u^{(l)})}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}}), \\ \mathbf{e}_i^{(l+1)} & =\sigma(\mathbf{e}_i^{(l)}+\sum_{u \in \mathcal{N}_i} \frac{\mathbf{e}_u^{(l)}+\phi_F(\mathbf{e}_u^{(l)} \odot \mathbf{e}_i^{(l)})}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}}).
\end{aligned}
```

**More datasets are yet to be tested, and this work is just a taste of whether KAN can be used for recommendation.**

## Structure

<img src="image/overview.pdf"/>

## Environment Requirement

- Python 3.9
- Pytorch 2.1.0

## Dataset

Two public datasets: MOOC, Games

## Training
  ```
  cd ./src
  python main.py
  ```
Thanks for simplifies [Recbole](https://github.com/RUCAIBox/RecBole) repo. [ImRec](https://github.com/enoche/ImRec).

## Citing if this repo. useful:

