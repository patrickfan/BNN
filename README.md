# BNN
Multimodel Ensemble predictions of Precipitation using Bayesian Neural Networks

We develop a Bayesian Neural Network (BNN) ensemble approach for large-scale precipitation predictions based on a set of CMIP6 climate models. BNN infers spatiotemporally varying model weights and biases through the calibration against observations. This ensemble scheme of BNN sufficiently leverages individual model skill for accurate predictions as well as provides interpretability about which models contribute more to the ensemble prediction at which locations and times to inform model development. Additionally, BNN accurately quantifies epistemic uncertainty to avoid overconfident projections. 


# Prerequisite

To run the code, make sure these packages are installed. This code has the following dependencies:

python >=3.6, 
tensorflow-gpu == 1.15,
matplotlib == 3.4.3,
numpy == 1.20.3,
scikit-learn == 0.24.2,
pandas == 1.3.4,
seaborn == 0.11.2

# Reference
Fan, Ming, Dan Lu, Deeksha Rastogi, and Eric M. Pierce. "A Spatiotemporal-Aware Weighting Scheme for Improving Climate Model Ensemble Predictions." Journal of Machine Learning for Modeling and Computing 3, no. 4 (2022). DOI: 10.1615/JMachLearnModelComput.2022046715
