# Anomaly Detection with Autoencoders
Semestral project for the Theory of Neural Networks at FIT CTU.

## Assignment
Comparison of multiple autoencoder architectures on the task of anomaly detection over Cardiotocogrpahy dataset (http://odds.cs.stonybrook.edu/cardiotocogrpahy-dataset/). The code for the experiments, along with additional comments can be found in the `outlier_detection.ipynb` Jupyter notebook.

The following autoencoder architectures were studied:
1. Undercomplete Autoencoder
2. Denoising Autoencoder
3. Variational Autoencoder


## Implementation

I implemented three PyTorch models, one for each autoencoder architecture. The second and third model were an extension of the first one, where the denoising autoencoder adds noise during training (as a regularization technique), and the variational autoencoder samples the latent vector `z` from a learnt distribution - my choice was the Normal distribution learned by two linear layers, one for Mu and the other for Sigma. 

The number and width of the hidden layers (as well as the size of the latent space) can be dynamically set in the models' constructors.

The hyperparameters are tuned in a custom grid search. Random seeds are set for reproducibility purposes. Then, for each architecture, models are trained with the hyperparameters that worked best given the metric (F1 score on the binary classification, positive class are the outliers). The results are shown in the table below.


## Results

| Architecture              |   F1 Score |   Accuracy |   Accuracy Inliers |   Accuracy Outliers |   Learning Rate |   Batch Size | Hidden Layers   |   Latent Space Size |   Seed |   MSE Threshold |
|:--------------------------|-----------:|-----------:|-------------------:|--------------------:|----------------:|-------------:|:----------------|--------------------:|-------:|----------------:|
| Undercomplete Autoencoder |   0.853868 |   0.899408 |           0.927492 |            0.846591 |           0.01  |           32 | [18]            |                  14 |      1 |             0.6 |
| Denoising Autoencoder     |   0.857955 |   0.901381 |           0.924471 |            0.857955 |           0.001 |           64 | [18, 16]        |                  12 |      0 |             4.6 |
| Variational Autoencoder   |   **0.861789** |   0.899408 |           0.897281 |            0.903409 |           0.001 |           32 | [18]            |                  12 |      2 |             1.7 |


## References
C. C. Aggarwal and S. Sathe, “Theoretical foundations and algorithms for outlier ensembles.” ACM SIGKDD Explorations Newsletter, vol. 17, no. 1, pp. 24–47, 2015.

Saket Sathe and Charu C. Aggarwal. LODES: Local Density meets Spectral Outlier Detection. SIAM Conference on Data Mining, 2016.