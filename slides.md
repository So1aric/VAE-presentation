---
theme: seriph
title: VAE
info: |
  A presentation for AI-intro class.
class: text-center
drawings:
  persist: false
transition: fade-out
overviewSnapshots: true
colorSchema: dark
---

# Variational Autoencoder

A simple but clever model to generate images.

---

# What is VAE?
<div />

In machine learning, a variational autoencoder (VAE) is an artificial neural network architecture introduced by Diederik P. Kingma and Max Welling. It is part of the families of probabilistic graphical models and variational Bayesian methods.
The encoder maps each point (such as an image) from a large complex dataset into a distribution within the latent space. The decoder has the opposite function, which is to map from the latent space to the input space, hence generating an image.

<img
  class="h-80"
  src="/pictures/VAE_Basic.png"
/>

---

# Formulation
<div />

<v-click>

For every datapoint $X$ in our dataset, we hope that there exists some latent variables which cause the model to generate something similar to $X$. Formally:

$$
f : \mathcal{Z} \times \Theta \to \mathcal{X}
$$

If $z$ is random and $\theta$ is fixed, then $f(z, \theta)$ is a random variable in the image space $\mathcal{X}$.

Usually we hope $z \sim \mathcal{N}(0, I)$ so that we could easily sample some $z$ and feed it in to the decoder to get an image.

</v-click>

<v-click>

We're aiming for a distribution $\theta$ which maximize $P_{\theta}(X)$.

$$
P_{\theta}(X) = \int_z P_{\theta}(X|z) P(z) \mathrm{d}z
$$

</v-click>

<v-click>

In VAE, the choice of the distribution $\theta$ is often Gaussian, i.e.,

$$
P_{\theta}(X|z) = P_{\mathcal{N}(f(z, \theta), \sigma^2 * I)}(X)
$$

</v-click>

---

# Formulation
<div />

<v-after>

For most $z$, $P_{\theta}(X|z)$ will be nearly zero. We want to sample $z$s that are likely to produce $X$.

</v-after>

<v-click>

This means that we need a new function $Q(z|X)$ that give us a distribution over $z$s.

</v-click>

<v-click>

We begin with Kullback-Leibler Divergence between $P(z|X)$ and $Q(z|X)$.

$$
\begin{align}

   \mathcal{D}[Q(z|X) \| P(z|X)]
&= \mathrm{E}_{z \sim Q}[\log Q(z|X) - \log P(z|X)] \\
&= \mathrm{E}_{z \sim Q}[\log Q(z|X) - \log \frac{P(z)P(X|z)}{P(X)}] \\
&= \mathrm{E}_{z \sim Q}[\log Q(z|X) - \log P(z) - \log P(X|z)] + \log P(X) \\

   \log P(X) - \mathcal{D}[Q(z|X) \| P(z|X)]
&= \mathrm{E}_{z \sim Q}[\log P(X|z)] - \mathcal{D}[Q(z|X) \| P(z)]

\end{align}
$$

</v-click>

<v-click>

The left hand side is something we want to maximize. The right hand side is something we could optimize.

</v-click>

---

# Formulation
<div />

The usual choice of $Q(z|X)$ is, again, Gaussian.

$$
Q(z|X) = \mathcal{N}(z|\mu(X), \Sigma(X))
$$

$\mu$ and $\Sigma$ are implemented via neural networks. Here, $\Sigma$ is constrained to be a diagonal matrix.

<v-click>

Thus, the term $\mathcal{D}[Q(z|X) \| P(z)]$ is now a KL-divergence between two multivariable Gaussian distributions.

$$
\mathcal{D}[\mathcal{N}(z|\mu(X), \Sigma(X)) \| \mathcal{N}(0, I)] = \frac 1 2 \big ( tr(\Sigma(X)) + \mu(X)^T\mu(X) - \log \det \Sigma(X) - n \big )
$$

</v-click>

<v-click>

$$
\begin{align}
   \mathrm{E}_{z \sim Q}[\log P(X|z)]
&= \mathrm{E}_{z \sim Q}[\log \mathcal{N}(X|f_{\theta}(z), I)] \\
&= \mathrm{E}_{z \sim Q}[\log \frac 1 {\sqrt{2\pi}} - \frac {\| X - f_{\theta}(z) \|_2^2} {2}]
\end{align}
$$

</v-click>

---

# Final Loss Function
<div />

<v-click>

Thus, we could construct the loss function:

$$
loss = \frac 1 2 \big( tr(\Sigma(X)) + \mu(X)^T\mu(X) - \log \det \Sigma(X) - n + \mathrm{E}_{z \sim Q}[\| X - f_{\theta}(z) \|_2^2] \big)
$$

</v-click>

<v-click>

We could interpret the loss function as the combination of encoder loss and decoder loss.

The KL-Divergence term is the encoder loss. It measures the encoder's ability at finding good $z$s.

The expectation term is the decoder loss. It measures the decoder's ability at reconstructing $X$ from $z$.

</v-click>

---

# Reparameterization
<div />

We have a loss function now, but it's not differentiable. Because $z$ is a random variable, we cannot backpropagate through it.

<v-click>

The reparameterization trick is that, instead of directly sample $z$ from $\mathcal{N}(\mu, \sigma^2)$, we first sample an $\epsilon$:

$$
\epsilon \sim \mathcal{N}(0, 1)
$$

</v-click>

<v-click>

Then we could calculate $z$:

$$
z = \mu + \sigma \epsilon
$$

Now $\mu$ and $\sigma$, which have learnable parameters, could be updated through the backpropagated gradient.

</v-click>

<v-click>

Recall: $\mu$,  $\sigma$ and $f_{\theta}(z)$ are neural networks.

</v-click>

---

# Training
<div />

- Given an $X$, feed it into two neural networks which give us $\mu$ and $\sigma$ respectively.
- Sample an $\epsilon$ from standard Gaussian $\mathcal{N}(0, I)$.
- Construct $z$ using $z = \mu + \sigma \epsilon$.
- Feed $z$ into a neural network $f_{\theta}(z)$.
- Calculate the total loss.
- Optimize the loss.

<v-click>

Note: the ouput of the $\sigma$ network could be adjusted to $\log \sigma^2$.

</v-click>

---
layout: two-cols
---

# Implementation
Using pytorch

```python {*}{maxHeight:'400px',class:'!children:text-0.6em'}
class VAE(nn.Module):
    def __init__(self, input_dim, inter_dim, latent_dim):
        super(CVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, inter_dim),
            nn.ReLU(),
            nn.Linear(inter_dim, latent_dim * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, inter_dim),
            nn.ReLU(),
            nn.Linear(inter_dim, input_dim),
            nn.Sigmoid(),
        )

    def sample(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x):
        org_size = x.size()
        batch = org_size[0]
        x = x.view(batch, -1)

        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        z = self.sample(mu, logvar)
        recon_x = self.decoder(z).view(size=org_size)

        return recon_x, mu, logvar

def loss(recon_x, x, mu, logvar):
  kl_loss = torch.sum(logvar.exp() + mu.pow(2) - logvar - 1)
  recon_loss = F.mse_loss(recon_x, x, reduction='sum')
  return kl_loss + recon_loss
```

::right::

<img
  src="/pictures/Learning_Curve.png"
  class="h-55"
/>

<img
  src="/pictures/Result.png"
  class="h-55"
/>

---
layout: cover
---

# Thanks!

