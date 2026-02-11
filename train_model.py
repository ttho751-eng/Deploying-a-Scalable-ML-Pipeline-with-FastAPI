import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

project_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(project_path, "data", "census_clean.csv")
model_path = os.path.join(project_path, "model", "model.pkl")
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
lb_path = os.path.join(project_path, "model", "lb.pkl")

print(data_path)
data = pd.read_csv(data_path)

train, test = train_test_split(data, test_size=0.20, random_state=42)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

model = train_model(X_train, y_train)

save_model(model, model_path)
save_model(encoder, encoder_path)
save_model(lb, lb_path)

preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# iterate through the categorical features
open("slice_output.txt", "w").close()
for col in cat_features:
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        p, r, fb = performance_on_categorical_slice(
            test,
            col,
            slicevalue,
            cat_features,
            "salary",
            encoder,
            lb,
            model,
        )
        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)
