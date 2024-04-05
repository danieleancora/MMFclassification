# MMFclassification

This small repository goes along the paper from Ancora et al., "Low-power multi-mode fiber projector overcomes shallow neural networks classifiers" 2024 and serves the purpose of reproducing Fig. 2.

To run these scripts you need to have installed RAPIDS.ai package, NUMPY, SCIPY, and MATPLOTLIB.

There are two files, the one that starts with 01classify is the actual run of the classification based on varius input datasets:
- standard MNIST
- zoomed MNIST
- randomized MNIST
- simulated speckles output with measured TM and MNIST as input
- measured speckles output after propagation through the MMF
This script will output the classification ACCURACY_ using each dataset.

The file that starts with 02plot_ is in charge of plotting all the results. Since the previous script require intense hardware computation, this one can load pretrained results to visualize them directly.

In principle, same results should be obtainable with sklearn.linear_model.LogisticRegression (from which the cuml.linear_model.LogisticRegression is derived) but this will require an extremely long time when output size is 600x600 as considered in the present case.
