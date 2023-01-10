# BankNote Authentication

- [BankNote Authentication](#banknote-authentication)
  - [Dataset Description](#dataset-description)
  - [Model Details](#model-details)

## Dataset Description

The [BankNote Authentication](https://archive.ics.uci.edu/ml/datasets/banknote+authentication) is a 
multivariate classification model that utilises simple logistic regression.

The dataset consists of 5 attributes:

1. Variance of Wavelet Transformed image (continuous)
2. Skewness of Wavelet Transformed image (continuous)
3. Curtosis of Wavelet Transformed image (continuous)
4. Entropy of image (continuous)
5. Class (integer)

Attributes 1 to 4 will be our input and the 5th attribute will be the output.  

## Model Details

- `Cost/Loss Function` &rarr; `Log Loss`
- `Activation Function` &rarr; `Sigmoid`
- `Optimizer` &rarr; `Batch Gradient Descent`


