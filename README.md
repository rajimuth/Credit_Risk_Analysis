# Credit_Risk_Analysis
## Purpose of the Analysis:
The goal of the analysis is to identify the best Machine Learning model that can be used to predict credit risk for loan applications. This will enable in selecting good candidates for loan which will help in low default rates.

Credit card dataset from a lending club was used for the current analysis. The data set will be sampled using different methods such as Random OverSampling, Undersampling using the Cluster Centroids Algorithm and the Combinatorial approach of over and undersampling using the SMOTEENN algorithm.

The resampled data will then be used to compare two new machine learning models BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk.



## Results:
- Initial task was to prepare the data  for machine learning by converting columns with text data into numerical data.
- This was done using the pd.get_dummies method from pandas

 <img width="557" alt="image" src="https://user-images.githubusercontent.com/94877067/165003344-361436d9-b3f7-4ae8-bdba-15695bd1afd5.png">

- The features and target were then created. The loan status was created as the target variable and all the other columns in the dataset was used as features.

 <img width="581" alt="image" src="https://user-images.githubusercontent.com/94877067/165003460-4913b222-048f-4d2d-8c77-6e54458fe7ed.png">
 
 - Next, the dataset is split into training and testing sets.

<img width="313" alt="image" src="https://user-images.githubusercontent.com/94877067/165003524-d419cf27-a3b7-44e9-a866-0bd33e17179c.png">

- The trained data was then scaled using Standard Scaler as the dataset had dispropotionate numerical values

<img width="447" alt="image" src="https://user-images.githubusercontent.com/94877067/165003828-ea975058-4c26-4e0c-9b9f-6be7f29aab92.png">

- COUNTER module shows the imbalance in  the training set 

<img width="328" alt="image" src="https://user-images.githubusercontent.com/94877067/165004087-b95c5a1d-c0b7-494c-8c9e-328fbf0dd341.png">


#### Sampling methods:
1) Oversampling:
- The minority class ('high_risk') is over sampled using  RandomOver Sampler
- Couter module after resampling shows same number if both the classes

<img width="422" alt="image" src="https://user-images.githubusercontent.com/94877067/165004248-8b5232c1-8800-4c4c-be9b-c1e00cbcd407.png">

-With a resampled dataset, Logistic Regression model was trained and used to make predictions. The models performance was also evaluated

- The Balanced accuracy score for Over Sampling is 0.485. The model was corrrect less than 50% of the time
- The precision for low risk is higher than high risk
- The recall(sensitivity) for predicting low_risk is very low than high_risk


<img width="437" alt="image" src="https://user-images.githubusercontent.com/94877067/165004578-3ff8ad10-ecd8-48e4-8740-432a9a1c44de.png">


<img width="464" alt="image" src="https://user-images.githubusercontent.com/94877067/165006743-42f6957e-d8e2-4bdf-aada-64ae9884c6be.png">

2) Undersampling:
- The majority class ('low_risk') is under sampled using  ClusterCentroids resampler
- Couter module after resampling shows same low number for both the classes

<img width="514" alt="image" src="https://user-images.githubusercontent.com/94877067/165006948-c4ff822c-6fca-428a-b814-23b5df4eea25.png">


-With a resampled dataset, Logistic Regression model was trained and used to make predictions. The models performance was also evaluated

- The Balanced accuracy score  is 0.485. The model was corrrect less than 50% of the time
- The precision for low risk is higher than high risk
- The recall(sensitivity) for predicting high_risk is very low than low_risk

<img width="535" alt="image" src="https://user-images.githubusercontent.com/94877067/165007048-07a0cc71-7471-426d-9891-e849a8ee99d7.png">


<img width="425" alt="image" src="https://user-images.githubusercontent.com/94877067/165007015-c14eec75-c893-4ba1-8e92-9fd617eaf11f.png">


3) Combination (Over and Under) Sampling:
- Combination sampling was performed using SMOTEENN technique
- Couter module after resampling shows similar  number for both the classes

<img width="425" alt="image" src="https://user-images.githubusercontent.com/94877067/165007167-81a68c49-6902-4ace-b0b6-af3df1b6aaab.png">


-With a resampled dataset, Logistic Regression model was trained and used to make predictions. The models performance was also evaluated

- The Balanced accuracy score for underSampling is 0.549. The model was corrrect more than 55% of the time
- The precision for low risk is higher than high risk
- The recall(sensitivity) for predictinglow_risk is very low than high_risk

<img width="376" alt="image" src="https://user-images.githubusercontent.com/94877067/165007263-69d81804-e971-41dd-83a8-5870c96d9310.png">


<img width="431" alt="image" src="https://user-images.githubusercontent.com/94877067/165007288-afd99800-9bc0-4276-9f90-4365b73d7a76.png">


##### Using Balanced Random Forest Classifier to make predictions on credit risk

- The Balanced accuracy score  is 0.5. The model was corrrect more than 50% of the time
- The precision for low risk (1) is very high than high risk (0)
- The recall(sensitivity) for predicting high_risk (0) is  low than low_risk (1)

The accuracy score, precison and recall were as shown below

<img width="419" alt="image" src="https://user-images.githubusercontent.com/94877067/165008219-698d0eec-8f8c-4fa3-a0b9-074307b09c6a.png">

##### Using Easy Ensemble AdabOOST Classifier to make predictions on credit risk
- The Balanced accuracy score  is 0.99. The model was correct more than 99% of the time
- The precision for low risk (1) is very low than high risk (0)
- The recall(sensitivity) for predicting high_risk was similar for both low (1) and high risk (0)


### SUMMARY:
In summary,with respect to accuracy score and recall score Easy Enseble Adaboost classifier faired well.  But the precision for high risk was also very high. This suggest that Adaboost clasifier can be a better model. But there is imbalance in the trained dataset as reflected by the counter module

<img width="366" alt="image" src="https://user-images.githubusercontent.com/94877067/165009856-994b41ae-4a46-4fc5-b9ef-459e284ff04c.png">

- The accuracy score was also too high that there can be a overfit. So it will be nice to have the model repeated after resampling the data preferably with the combination of Over and Under Sampling as the scores were better in combination with the logistic regression model. 







