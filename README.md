# Deploying a Scalable ML Pipeline with FastAPI

This project was completed as part of the Udacity Machine Learning DevOps Engineer Nanodegree.

The goal of this project was to develop a classification model, write code to monitor performance on data slices, and deploy model with FastAPI package.
The process included:
- Data versioning with DVC
- Model training and artifact persistence
- Slice-based model performance evaluation
- Unit testing with pytest
- Code linting with flake8
- Continuous Integration using GitHub Actions
- Deployment of a REST API using FastAPI

## Project Repository

GitHub Repository:
https://github.com/ttho751-eng/Deploying-a-Scalable-ML-Pipeline-with-FastAPI

## Model Overview

The model is a Logistic Regression classifier trained on the Census Income dataset to predict whether an individual's income exceeds $50K per year.

Artifacts saved:
- `model/model.pkl`
- `model/encoder.pkl`
- `model/lb.pkl`

## API

The FastAPI application provides:

- GET `/` — Returns a welcome message.
- POST `/data/` — Performs model inference on input data.

## Testing and CI

- Unit tests implemented in `test_ml.py`
- Linting with flake8
- GitHub Actions runs pytest and flake8 on push to main

Screenshots of CI passing and API testing are included in the `screenshots/` directory.

