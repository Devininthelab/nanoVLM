# Notes 
## RMS Norm

The Root Mean Square (RMS) Norm of a vector $x \in \mathbb{R}^n$ is defined as $\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2 + \epsilon}}$, where $\epsilon$ is a small constant added for numerical stability. Given a vector $x = [x_1, x_2, \ldots, x_n]$, the RMS Norm is computed by first calculating the mean of the squares of the elements, adding a small constant $\epsilon$, and then taking the square root of this sum. The vector is then normalized by dividing each element by this value.


## Rotary Embeddings

Rotary embeddings are a method of encoding positional information in transformer models. Rope = relative positional encoding + kinda sin/cos thing.


RoPE (Rotary Position Embedding) applies a position-dependent rotation to the query and key vectors:

Given a query vector $\mathbf{q} = [q_0, q_1, \ldots, q_{d-1}]$ and a key vector $\mathbf{k} = [k_0, k_1, \ldots, k_{d-1}]$, the rotary embedding applies a rotation based on the position $m$ of the token in the sequence:
$$\mathbf{R}_{\Theta,m}\mathbf{q} = [q_0, q_1, \ldots, q_{2i}\cos(m\theta_i) - q_{2i+1}\sin(m\theta_i), q_{2i}\sin(m\theta_i) + q_{2i+1}\cos(m\theta_i), \ldots]$$

Where:
- $\theta_i = 10000^{-2i/d}$ (frequency parameters)
- $m$ is token position
- $d$ is embedding dimension

This rotation preserves inner products between query and key vectors while encoding their relative positions.

## Group Query Attention
Instead of computing attention for each query independently, group query attention computes attention for groups of queries together. This reduces the number of attention computations and can lead to more efficient training and inference. For example, if you have $k$ queries, you can group them into $g$ groups, where each group contains $k/g$ queries. The attention is then computed for each group, and the results are aggregated.