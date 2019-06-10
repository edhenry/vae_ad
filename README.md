# vae_ad
Repository containing vanilla variational autoencoder for anomaly detection.

## Autoencoding Variational Bayes

This repository is an extension of the work done by Kingma et al. in [Autoencoding Variational Bayes](http://arxiv.org/abs/1312.6114)

The idea is to use this approach as a way to model, probabalistically, that variational autoencoders can be used for novelty detection given that the encoder/decoder architecture is finding a latent set of codes (z) that are representative of the underlying data generating distribution. This repository is using MNIST as an example, but the concept still applies to many other domains that have a continuous representations as input.
