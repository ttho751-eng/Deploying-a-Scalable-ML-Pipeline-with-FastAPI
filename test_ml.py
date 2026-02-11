import numpy as np

from ml.model import train_model, inference, compute_model_metrics


def test_one():
    """Test that train_model returns a trained model object."""
    X = np.array([[0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1])

    model = train_model(X, y)

    assert model is not None
    assert hasattr(model, "predict")


def test_two():
    """Test that compute_model_metrics returns three floats."""
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


def test_three():
    """Test that inference returns predictions of correct length."""
    X = np.array([[0, 1], [1, 0]])
    y = np.array([0, 1])

    model = train_model(X, y)
    preds = inference(model, X)

    assert len(preds) == len(X)
