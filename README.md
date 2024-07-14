# NMT-Project
## About this project
Machine Translation is one of the most important problems in natrual language processing. In this project, using Transformer model, we acheived 49 BLEU score on translating English to Persian sentences.

## Introduction
Machine Translation is a sequence-to-sequence problem; in other words, it requires a sequence as input and returns a sequence as output. These kind of problems are often solved with Encoder-Decoder models. Encoder takes a sequence as input, generates some vectors that represent the input then decoder uses those vectors to generate the output via decoding them.

In Encoder and Decoder, LSTM units might be used to store longer sequences.

### Attention Mechanism
One of the issues with Simple Encoder-Decoder models that use LSTM, is that they need to represent any input in a finite space. For example let's say we have a sentence like *I really love AI* given to the Encoder to encode it to $\mathbb{R}^{64}$; now imagine a book with 1000 pages is given to the encoder. No matter what the input is, encoder is supposed to summerize is into a fixed space.

To address this issue we can return a vector at each time step. In this case we have the representation of the whole sentence at each time but focusing more on the current time-step. The decision for what parts the decoder should attend to at each time-step is made via decoder's hidden state.

Attention mechanism gives weight to each encoded time-step as to which part is should attend to more. The result of this is a weighted average of the encoder's output.

Attention is calculated via computing the similarity. Attention for the *RNN with Attention* part of the project uses **additive compatibility function** also known as **additive attention** and is calculated as such:
$$
\begin{equation}
    a(s_{i-1}, h_j) = v_a^\top tanh(W_a s_{i-1} + U_ah_j)
\end{equation}
$$

where $s_i$ is the decoder's hidden state at time-step $i$ and $h_j$ is the encoders output at time-step $j$, $W_a \in \mathbb{R}^{n \times n}$ and $U_a \in \mathbb{R}^{n\times 2n}$ (Because we are using BiRNN for the encoder).

Also $v_a \in \mathbb{R}^n$ which projects [Seqeuence Length, n] to [Sequence Length, 1] which we denote energies for the attention.

Context to be given to the decoder is calculated as the weighted sum of the encoder outputs:
$$
\begin{equation}
    c_i = \sum_{j=1}^{T_x} \alpha_{ij}h_j
\end{equation}
$$
where $c_i$ is context at time-step $i$ and $T_x$ is length of the encoder's sequence.
$$
\begin{align}
    \alpha_{ij} &= \frac{exp(e_{ij})}{\sum_{k=1}^{T_x}exp(e_{ik})}\\
    e_{ij} &= v_a^\top tanh(W_a s_{i-1} + U_ah_j)
\end{align}
$$

We can show that, how much attention the decoder is giving to each sequence of the encoder via $\alpha_i$

### RNN
Although we have solve the issue with Encoder-Decoder models with attention, we cannot benefit much from parallelism since RNN units are sequential.

Here is how GRU is calculated:
$$
\begin{align}
    \tilde{h}_t &= tanh(W_c x_t +U_c[h_{t-1} \odot r_t] + b_c)\\
    r_t &= \sigma(W_r x_t + U_r h_{t-1} + b_r)\\
    z_t &= \sigma(W_z x_t + U_z h_{t-1} + b_z)\\
    h_i &= z_t \odot \tilde{h}_t + (1 - z_t)h_{t-1}\\
\end{align}
$$

It can be seen the recurrent relation is embedded in the calculations.


### MORE DOCUMENTATION WILL BE WRITTEN SOON