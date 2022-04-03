# Model Card
Binary Classification Model to predict income class ( >= 50k or < 50k) from given census data like (age, workclass, education etc.).

## Model Details
GradientBoosting - Classifiers from [Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) with selected hyperparamerters as below list 
``` python
- n_estimators: 100
- learning_rate: 0.05
- min_samples_split: 500
- min_samples_leaf: 50
- max_depth: 8
```
## Intended Use
Use for binary classification problem on [Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/census+income). <br/>
 This model will classify whether from given census data, it will have income higher than 50k (= 1) of lower and equal 50k (= 0)
## Training Data
Data for model development come from [Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/census+income). This data set compose with 14 attibutes mixed with categorical data and numerical data. <br/>
For training dataset, 80% of data have been randomly selected.
## Evaluation Data
Data for evalution is the same as training set, but remaining 20% of data is used for evaluation by slicing on each categorical features data.
## Metrics
Selected performance metrics is ```accuracy``` on KFold cross-validation with n = 10 on trainining data. <br/>
```accuracy_mean``` = 0.831 and ```accuracy_std``` = 0.06
## Ethical Considerations
Original data contains ```race```, ```gender```, ```education``` and ```native_country``` which all of them is highly skewed, especially ```native_country``` that mostly come this **USA**. This will drive the model to discriminate people base on these demographic and geographic features set.<br/>
Investigation before using it should be done.

## Caveats and Recommendations
