import requests
import numpy as np

BASE_URL = "http://localhost:8000/api/v1"

def test_regression():
    print("Testing Linear Regression...")
    model_id = "test_linear"
    # Simple linear data: y = 2x + 1
    X = [[1.0], [2.0], [3.0], [4.0]]
    y = [3.0, 5.0, 7.0, 9.0]
    
    # Train
    train_res = requests.post(
        f"{BASE_URL}/train",
        json={
            "model_id": model_id,
            "config": {"model_type": "linear_regression"},
            "X": X,
            "y": y
        }
    )
    print(f"Train Status: {train_res.status_code}, Resp: {train_res.json()}")
    
    # Predict
    predict_res = requests.post(
        f"{BASE_URL}/predict",
        json={
            "model_id": model_id,
            "X": [[5.0]]
        }
    )
    print(f"Predict Status: {predict_res.status_code}, Resp: {predict_res.json()}")
    assert predict_res.status_code == 200
    pred = predict_res.json()["predictions"][0]
    print(f"Prediction for 5.0: {pred} (Expected ~11.0)")

def test_classification():
    print("\nTesting Logistic Regression...")
    model_id = "test_logistic"
    # Simple binary classification
    X = [[0.1], [0.2], [0.8], [0.9]]
    y = [0, 0, 1, 1]
    
    # Train
    train_res = requests.post(
        f"{BASE_URL}/train",
        json={
            "model_id": model_id,
            "config": {"model_type": "logistic_regression"},
            "X": X,
            "y": y
        }
    )
    print(f"Train Status: {train_res.status_code}, Resp: {train_res.json()}")
    
    # Predict
    predict_res = requests.post(
        f"{BASE_URL}/predict",
        json={
            "model_id": model_id,
            "X": [[0.15], [0.85]]
        }
    )
    print(f"Predict Status: {predict_res.status_code}, Resp: {predict_res.json()}")
    preds = predict_res.json()["predictions"]
    print(f"Predictions: {preds} (Expected [0, 1])")

def test_clustering_kmeans():
    print("\nTesting KMeans...")
    model_id = "test_kmeans"
    # Clusters around 0 and 10
    X = [[0.1], [0.2], [9.8], [9.9], [10.1]]
    
    # Train
    train_res = requests.post(
        f"{BASE_URL}/train",
        json={
            "model_id": model_id,
            "config": {
                "model_type": "kmeans",
                "params": {"n_clusters": 2, "n_init": 10}
            },
            "X": X
        }
    )
    print(f"Train Status: {train_res.status_code}, Resp: {train_res.json()}")
    
    # Predict
    predict_res = requests.post(
        f"{BASE_URL}/predict",
        json={
            "model_id": model_id,
            "X": [[0.15], [10.0]]
        }
    )
    print(f"Predict Status: {predict_res.status_code}, Resp: {predict_res.json()}")
    preds = predict_res.json()["predictions"]
    print(f"Predictions: {preds} (Expected distinct clusters)")

def test_dpc():
    print("\nTesting DPC (Custom)...")
    model_id = "test_dpc"
    X = [[1,1], [1,2], [2,1], [10,10], [10,11], [11,10]]
    
    # Train
    train_res = requests.post(
        f"{BASE_URL}/train",
        json={
            "model_id": model_id,
            "config": {"model_type": "dpc", "params": {"dc": 2.0}},
            "X": X
        }
    )
    print(f"Train Status: {train_res.status_code}, Resp: {train_res.json()}")
    
    # Predict
    predict_res = requests.post(
        f"{BASE_URL}/predict",
        json={
            "model_id": model_id,
            "X": [[1.5, 1.5]]
        }
    )
    print(f"Predict Status: {predict_res.status_code}, Resp: {predict_res.json()}")
    preds = predict_res.json()["predictions"]
    print(f"Predictions: {preds}")

if __name__ == "__main__":
    try:
        test_regression()
        test_classification()
        test_clustering_kmeans()
        test_dpc()
    except Exception as e:
        print(f"Verification failed: {e}")
