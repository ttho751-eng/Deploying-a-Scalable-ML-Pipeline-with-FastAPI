# Model Card

## Model Details:
The model is a binary classification Logistic Regression model developed by a student developer as part of the Udacity Machine Learning DevOps Engineer Nanodegree program. The model was trained to predict whether an 
individual’s income exceeds $50K using the Census Income dataset. It uses a mix of categorical and numerical features. The categorical variables were processed through one-hot encoding and the target label binarized for 
training. The model was trained using an 80/20 train-test split and evaluated using precision, recall, and F1 score.There was an additional slice-based evaluation to assess subgroup performance. Trained model and 
preprocessing artifacts were saved for reuse during inference and deployment to avoid retraining the model. 

## Intended Use:
The primary use would be educational demonstration of an end-to-end MLOps pipeline, including data preprocessing, model training, evaluation (with slice metrics), persistence, and deployment via a REST API. via a REST API.
Intended users are Udacity reviewers, students, and developers learning ML deployment workflows.
Out-of-scope use would be any real-world or high-stakes decision-making such as hiring, healthcare, or legal decisions.

## Training Data:
The model was trained using the UCI Census Income dataset provided as census.csv.
The target variable for the model is salary, which represents a binary classification outcome of earning less than or equal to $50K versus more than $50K.
A cleaned version of the dataset was created by removing spaces from the raw CSV file to address formatting issues.
Categorical features were processed using one-hot encoding.The target labels were binarized using a LabelBinarizer.
The categorical features used in training include workclass, education, marital-status, occupation, relationship, race, sex, and native-country.
lBinarizer.

## Evaluation Data:
The dataset was split into training and testing sets using a train/test split.
Twenty percent of the dataset was reserved for evaluation, while eighty percent was used for training.
A random state of 42 was used to ensure reproducibility.
Model performance was evaluated on the held-out test set, and additional evaluations were conducted on slices of the data across categorical features.

## Metrics:
The model was evaluated using standard binary classification metrics, including precision, recall, and F1 score.
On the test dataset, the model achieved a precision score of 0.7318.
The recall score on the test dataset was 0.5646.
The resulting F1 score was 0.6374.
Model performance was also evaluated across categorical data slices. With detailed results written to slice_output.txt.
For example, when evaluating the slice where workclass was Federal-gov, the model achieved a precision of 0.07377, a recall of 0.6429, and an F1 score of 0.687.
Slice-based evaluation was used to help identify potential disparities in model performance across different subgroups.

## Ethical Considerations:
This model was trained on historical census data that includes sensitive demographic attributes such as sex, race, and native country. Because of this, the model may reflect historical patterns 
and societal biases present in the underlying dataset. Although slice-based performance evaluation was performed to better understand how the model behaves across different subgroups, this analysis 
alone does not guarantee fairness or eliminate bias.
The model is intended solely for educational and demonstration purposes as part of an MLOps workflow. It should not be used to make real-world decisions that could affect individuals’ employment, 
financial opportunities, housing, healthcare, or legal outcomes.
Any deployment of similar models in real applications would require deeper fairness assessments, stakeholder review, and ongoing monitoring. 

## Caveats and Recommendations
During training, a convergence warning was observed when fitting the Logistic Regression model, indicating that the maximum number of iterations was reached. While the model successfully trained 
and produced stable evaluation metrics, future iterations could improve convergence by increasing the maximum number of training iterations or applying feature scaling.
Model performance may vary when applied to data that differs from the training distribution, particularly for underrepresented or noisy categorical values such as unknown or missing entries. 
Slice-based evaluation highlighted that certain subgroups exhibit lower recall or F1 scores, which suggests that additional investigation may be warranted before deployment.
Future improvements could include improved handling of missing or unknown categories and more comprehensive bias and fairness evaluations. 
Additional monitoring and periodic retraining would also be recommended if the model were adapted for any real-world use case.
