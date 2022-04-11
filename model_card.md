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
**Train Set Performance** <br/>
```accuracy_mean``` = 0.842 and ```accuracy_std``` = 0.006

**Test Set Performance** <br/>
```accuracy_mean``` = 0.838 and ```accuracy_std``` = 0.012

For full test data set, precision, recall, fbeta have been calculated. <br/>
**Train Set** <br/>
```Precision``` = 0.735, ```Recall``` = 0.611, ```FBeta``` = 0.667 <br/>
**Test Set** <br/>
```Precision``` = 0.711, ```Recall``` = 0.549, ```FBeta``` = 0.620

## Ethical Considerations
Original data contains ```race```, ```gender```, ```education``` and ```native_country``` which all of them is highly skewed, especially ```native_country``` that mostly come this **USA**. This will drive the model to discriminate people base on these demographic and geographic features set.<br/>
Investigation before using it should be done.

## Caveats and Recommendations
This model is selected base on trial and error only, hyperparameter haven't yet perform tuning and also model type have been selected for learning purpose only. Fine tuning on both hyperparameter and model selection should be done before using.

Also carefully investigate on [slice_output.txt](".model/slice_output.txt), performance base on specific features is vary. So, it need to carefullly check on contexual of your data.