from sklearn.metrics import classification_report

def evaluate_model(model, test_data, test_labels):
    predictions = model.predict(test_data)
    predicted_labels = np.argmax(predictions, axis=1)
    print(classification_report(test_labels, predicted_labels))