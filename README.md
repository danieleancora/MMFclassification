# MMFclassification

This small repository goes along the paper from Ancora et al., "Low-power multi-mode fiber projector overcomes shallow neural networks classifiers" 2024 and serves the purpose of reproducing Fig. 2.

To run these scripts you need to have installed the RAPIDS.ai package, including NUMPY, CUPY, SCIPY, and MATPLOTLIB.

There are six script files. 

The ones that starts with `0Xclassify_` are the actual routines to do classification based on varius input datasets:
- standard MNIST
- randomized MNIST
- zoomed MNIST
- measured speckles output after propagation through the MMF
- simulated speckles output with measured TM and MNIST as input

Each of these scripts will output the classification file named `ACCURACY_` using each dataset. To run these scripts on your own machine, you need to download the datasets and put them in the same folder where these scripts are. 
The datasets can be downloaded at the following DOI: 10.6084/m9.figshare.25551186


The file that starts with `06plot_` is in charge of plotting all the results. Since the previous scripts require intense hardware computation, this one can load pretrained results to visualize them directly. In this example, we rerun 50 independent optimizations of all the datasets considered by varying the number of training images in steps of 500. We may notice the same trend, being the experimental dataset the one providing the highest performance among the others.


EXECUTION NOTE: In principle, same results should be obtainable with `sklearn.linear_model.LogisticRegression` (from which the `cuml.linear_model.LogisticRegression` is derived) but this will require an extremely long time when output size is 600x600 as considered in the present case.


