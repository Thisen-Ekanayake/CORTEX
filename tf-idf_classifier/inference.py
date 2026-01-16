import os
import joblib

LABEL_MAP = {
    "chat": 0,
    "meta": 1,
    "rag": 2,
}

# Reverse mapping for output
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


class TextClassifier:
    def __init__(self, model_dir="tf-idf_classifier/model"):
        """Load the trained vectorizer and classifier."""
        vectorizer_path = os.path.join(model_dir, "tfidf.joblib")
        classifier_path = os.path.join(model_dir, "classifier.joblib")
        
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Classifier not found: {classifier_path}")
        
        self.vectorizer = joblib.load(vectorizer_path)
        self.classifier = joblib.load(classifier_path)
        print(f"Model loaded from {model_dir}")
    
    def predict(self, text):
        """
        Predict the category and confidence scores for a given text.
        
        Args:
            text (str): Input text to classify
            
        Returns:
            dict: Dictionary containing predicted label and confidence scores
        """
        # Transform the input text
        X = self.vectorizer.transform([text])
        
        # Get prediction probabilities
        probabilities = self.classifier.predict_proba(X)[0]
        
        # Get the predicted class
        predicted_class = self.classifier.predict(X)[0]
        predicted_label = ID_TO_LABEL[predicted_class]
        
        # Create confidence scores dictionary
        confidence_scores = {
            ID_TO_LABEL[i]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        return {
            "predicted_label": predicted_label,
            "confidence_scores": confidence_scores,
            "predicted_class_id": int(predicted_class)
        }
    
    def predict_batch(self, texts):
        """
        Predict categories for multiple texts.
        
        Args:
            texts (list): List of input texts
            
        Returns:
            list: List of prediction dictionaries
        """
        X = self.vectorizer.transform(texts)
        probabilities = self.classifier.predict_proba(X)
        predictions = self.classifier.predict(X)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            confidence_scores = {
                ID_TO_LABEL[j]: float(prob) 
                for j, prob in enumerate(probs)
            }
            results.append({
                "text": texts[i],
                "predicted_label": ID_TO_LABEL[pred],
                "confidence_scores": confidence_scores,
                "predicted_class_id": int(pred)
            })
        
        return results


def print_results(result):
    """Pretty print the classification results."""
    print("\n" + "="*50)
    print(f"Predicted Category: {result['predicted_label'].upper()}")
    print("="*50)
    print("\nConfidence Scores:")
    print("-"*50)
    
    # Sort by confidence (descending)
    sorted_scores = sorted(
        result['confidence_scores'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    for label, score in sorted_scores:
        bar_length = int(score * 40)
        # bar = "█" * bar_length + "░" * (40 - bar_length) {bar}
        print(f"{label:>6}: {score:.2%}")
    print("="*50 + "\n")


if __name__ == "__main__":
    # Initialize the classifier
    classifier = TextClassifier(model_dir="tf-idf_classifier/model")
    
    # Example usage with single prediction
    test_prompts = [
        "What's the weather like today?",
        "How do I change the model settings?",
        "Can you search the database for customer information?",
        "Tell me a joke",
        "What tools do you have access to?",
        "Find documents related to Q4 sales report"
    ]
    
    print("\nSingle Prediction Examples:")
    for prompt in test_prompts:
        print(f"\nInput: '{prompt}'")
        result = classifier.predict(prompt)
        print_results(result)
    
    """# Example usage with batch prediction
    print("\nBatch Prediction Example:")
    batch_results = classifier.predict_batch(test_prompts)
    for result in batch_results:
        print(f"\nText: '{result['text']}'")
        print(f"Category: {result['predicted_label']} ({result['confidence_scores'][result['predicted_label']]:.2%})")"""


# Simple usage
# from inference import TextClassifier
#
# classifier = TextClassifier(model_dir="tf-idf_classifier")
# result = classifier.predict("What's the weather today?")
#
# print(result['predicted_label'])  # e.g., "chat"
# print(result['confidence_scores'])  # {'chat': 0.85, 'meta': 0.10, 'rag': 0.05}