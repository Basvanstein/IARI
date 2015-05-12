# IARI - Incremental Attribute Regression Imputation

Real-life datasets that occur in domains such as industrial process control, medical diagnosis, marketing, risk management, often contain missing values. This poses a challenge for many classification and regression algorithms which require complete training sets. We present a new approach for "repairing" such incomplete datasets by constructing a sequence of regression models that iteratively replace all missing values. Additionally, our approach uses the target attribute to estimate the values of missing data. The accuracy of our method, Incremental Attribute Regression Imputation, IARI, is compared with the accuracy of several popular and state of the art imputation methods, by applying them to five publicly available benchmark datasets. The results demonstrate the superiority of our approach.

## HOW TO USE

To run the IARI algorithm you need Python 2.7.X, NumPi, Scikit-Learn and a few other python packages installed. 
Some of the datasets are provided in this repository and should be in the same folder as the script, other datasets will be downloaded on the fly.

Run the algorithm with the command: *python test.py datasetname modelname > outputfile.txt*
Where datasetname can be one of: page, concrete, digits, iris, cover, or allhouses.
And modelname can be one of: Gradient, RandomForest, SVM, or Gaussian.

Note that you need the folder "img" and "arrays" to store the results.
The final results will be outputted in a latex table format.

## Results

You can find all the results of all our experiments together with the experimental setup here: [Results.pdf](https://github.com/Basvanstein/IARI/blob/master/AlgorithmResults.pdf)
