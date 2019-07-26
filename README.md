# SpectralNormalization-TF-Keras

Transfer ["IShengFang/SpectralNormalizationKeras"](https://github.com/IShengFang/SpectralNormalizationKeras) to tf.keras

I created this repo, because Keras and tf.keras are not compatible, and some libs of Keras are not supported by old Tensorflow versions.

Enviro: Tensorflow 1.12.0

# How to use?

1. Move SNorm_tf_keras.py to your code's dir.
2. Import layers like  
`from SpectralNormalizationKeras import DenseSN, ConvSN1D, ConvSN2D, ConvSN3D`
3. Use them like normal layers (but only in discriminators mentioned in the paper).

# Does this method work?

[View here](https://github.com/IShengFang/SpectralNormalizationKeras#result)

---

*[original paper on arxiv](https://arxiv.org/abs/1802.05957)*
