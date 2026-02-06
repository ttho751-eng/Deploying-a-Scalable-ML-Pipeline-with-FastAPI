# Model Card

## Model Details

Developed by: Tarynn Thompson, as part of an academic project for the Udacity Machine Learning DevOps Engineer Nanodegree.

Model type: Binary classification model

Algorithm: Logistic Regression (scikit-learn)

Objective: Predict whether an individual’s income exceeds $50K/year

Model artifacts:
model/model.pkl – trained classifier
model/encoder.pkl – fitted OneHotEncoder for categorical features
model/lb.pkl – fitted LabelBinarizer for the target label

Training environment: Python (virtual environment), scikit-learn

## Intended Use
Primary use: Educational demonstration of an end-to-end MLOps pipeline, including data preprocessing, model training, evaluation (with slice metrics), persistence, and deployment via a REST API.

Intended users: Udacity reviewers, students, and developers learning ML deployment workflows.

Out-of-scope use: Any real-world or high-stakes decision-making (e.g., hiring, lending, housing, healthcare, or legal decisions).

## Training Data
Dataset: UCI Census Income (Adult) dataset (census.csv)

Target variable: salary (binary classification: <=50K vs >50K)

Data preprocessing:

A cleaned dataset (census_clean.csv) was created by removing spaces from the raw CSV to address formatting issues.

Categorical features were one-hot encoded.

Labels were binarized using a LabelBinarizer.

Categorical features used:

workclass

education

marital-status

occupation

relationship

race

sex

native-country
## Evaluation Data
Data split: Train/test split

Test size: 20% of the dataset

Random state: 42

Evaluation approach:

Overall metrics computed on the held-out test set.

Additional performance evaluation performed on slices of the test data across categorical features.

## Metrics
The model was evaluated using standard binary classification metrics:

Precision: 0.7318

Recall: 0.5646

F1 Score: 0.6374

Slice Metrics

Performance was also evaluated across categorical data slices, with results written to slice_output.txt.

Example slice:

workclass = ? (Count: 389)

Precision: 0.7500

Recall: 0.2143

F1: 0.3333

Slice-based evaluation helps identify potential disparities in model performance across subgroups.

## Ethical Considerations
This model was trained on historical census data that includes sensitive demographic attributes such as sex, race, and native country. As a result, the model may reflect historical patterns and societal biases present in the underlying dataset. Although slice-based performance evaluation was performed to better understand how the model behaves across different subgroups, this analysis alone does not guarantee fairness or eliminate bias.

The model is intended solely for educational and demonstration purposes as part of an MLOps workflow. It should not be used to make real-world decisions that could affect individuals’ employment, financial opportunities, housing, healthcare, or legal outcomes. Any deployment of similar models in real applications would require deeper fairness assessments, stakeholder review, and ongoing monitoring.

## Caveats and Recommendations
During training, a convergence warning was observed when fitting the Logistic Regression model, indicating that the maximum number of iterations was reached. While the model successfully trained and produced stable evaluation metrics, future iterations could improve convergence by increasing the maximum number of training iterations or applying feature scaling.

Model performance may vary when applied to data that differs from the training distribution, particularly for underrepresented or noisy categorical values such as unknown or missing entries. Slice-based evaluation highlighted that certain subgroups exhibit lower recall or F1 scores, which suggests that additional investigation may be warranted before deployment.

Future improvements could include hyperparameter tuning, improved handling of missing or unknown categories, and more comprehensive bias and fairness evaluations. Additional monitoring and periodic retraining would also be recommended if the model were adapted for any real-world use case.
