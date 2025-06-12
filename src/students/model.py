import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np
import pandas as pd
from transformers import RobertaModel, RobertaTokenizer, GPT2Model, GPT2Tokenizer, AutoModel, AutoTokenizer
import re

class RobertaClassifier(nn.Module):
    def __init__(self, num_labels=2):
        super(RobertaClassifier, self).__init__()
        self.roberta_model = RobertaModel.from_pretrained("roberta-large-openai-detector")
        self.classifier = nn.Sequential(
            nn.Linear(self.roberta_model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(256, num_labels)
        )

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.mean_pooling(roberta_output, attention_mask)
        logits = self.classifier(pooled_output)
        return logits

class GPT2Classifier(nn.Module):
    def __init__(self, num_labels=2):
        super(GPT2Classifier, self).__init__()
        self.gpt2_model = GPT2Model.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        self.gpt2_model.resize_token_embeddings(len(self.tokenizer))

        self.classifier = nn.Sequential(
            nn.Linear(self.gpt2_model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2_model(input_ids=input_ids, attention_mask=attention_mask)
        last_token = outputs.last_hidden_state[:, -1, :]
        logits = self.classifier(last_token)
        return logits

class SimilarityClassifier_zero(nn.Module):
    def __init__(self, num_labels=2, freeze_ratio=0.82):
        super(SimilarityClassifier_zero, self).__init__()

        # Load DeBERTa Model
        self.deberta_model = AutoModel.from_pretrained("MoritzLaurer/deberta-v3-large-zeroshot-v2.0")

        num_layers = len(self.deberta_model.encoder.layer)
        freeze_layers = int(num_layers * freeze_ratio)

        for layer in self.deberta_model.encoder.layer[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

        # Classifier Head (2 outputs for binary classification)
        self.classifier = nn.Sequential(
            nn.Linear(self.deberta_model.config.hidden_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),     
            nn.Dropout(0.35),
            nn.Linear(256, num_labels) 
        )

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        deberta_output = self.deberta_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.mean_pooling(deberta_output, attention_mask)
        logits = self.classifier(pooled_output)
        return logits 

class SimilarityClassifier_cross(nn.Module):
    def __init__(self, num_labels=2, freeze_ratio=0.70):
        super(SimilarityClassifier_cross, self).__init__()

        # Load DeBERTa Model
        self.deberta_model = AutoModel.from_pretrained("cross-encoder/nli-deberta-v3-large")

        # Freeze a portion of the layers
        num_layers = len(self.deberta_model.encoder.layer)
        freeze_layers = int(num_layers * freeze_ratio)

        for layer in self.deberta_model.encoder.layer[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

        # Classifier Head (2 outputs for binary classification)
        self.classifier = nn.Sequential(
            nn.Linear(self.deberta_model.config.hidden_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),     
            nn.Dropout(0.2),
            nn.Linear(256, num_labels) 
        )

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        deberta_output = self.deberta_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.mean_pooling(deberta_output, attention_mask)
        logits = self.classifier(pooled_output)
        return logits 

# Initialize tokenizers and models
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-large-openai-detector")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load AI detection models
roberta_model = RobertaClassifier().to(device)
gpt2_model = GPT2Classifier().to(device)

# Load  trained weights
roberta_model.load_state_dict(torch.load("model_w_robeta_v1.pkl", map_location=device))
gpt2_model.load_state_dict(torch.load("model_w_gpt.pkl", map_location=device))

models = [roberta_model, gpt2_model]
tokenizers = [roberta_tokenizer, gpt2_tokenizer]

# Load similarity models
tokenizer_zero = AutoTokenizer.from_pretrained("MoritzLaurer/deberta-v3-large-zeroshot-v2.0")
tokenizer_cross = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-large")

similarity_models = []
similarity_tokenizers = []

# Load similarity model - zero shot version
try:
    model_zero = SimilarityClassifier_zero().to(device)
    model_zero.load_state_dict(torch.load("similarity_model____1.pkl", map_location=device))
    model_zero.eval()
    similarity_models.append(model_zero)
    similarity_tokenizers.append(tokenizer_zero)
    print("Successfully loaded zero-shot similarity model")
except Exception as e:
    print(f"Error loading zero-shot similarity model: {e}")

# Load similarity model - cross encoder version
try:
    model_cross = SimilarityClassifier_cross().to(device)
    model_cross.load_state_dict(torch.load("similarity_model_________1442.pkl", map_location=device))
    model_cross.eval()
    similarity_models.append(model_cross)
    similarity_tokenizers.append(tokenizer_cross)
    print("Successfully loaded cross-encoder similarity model")
except Exception as e:
    print(f"Error loading cross-encoder similarity model: {e}")

try:
    ai_detection_meta_model = joblib.load("logistic_regression_model_AI.pkl")
    print("Successfully loaded AI detection meta model")
except Exception as e:
    print(f"Error loading AI detection meta model: {e}")
    ai_detection_meta_model = None

try:
    similarity_meta_model = joblib.load("logistic_regression_model.joblib")
    print("Successfully loaded similarity meta model from logistic_regression_model.joblib")
except Exception as e:
    print(f"Error loading similarity meta model: {e}")
    similarity_meta_model = None

roberta_model.eval()
gpt2_model.eval()

def clean_text(text):
    text = text.strip() 
    text = ' '.join(text.split())  
    return text

def predict_ai_detection_logistic(models, ai_meta_model, text, tokenizers, max_length=256, device=None, threshold=0.7):
    """
    Predict AI-generated text using separate logistic model for AI detection
    """
    if not clean_text(text):
        return None, 0.0

    text = clean_text(text)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if ai_meta_model is None:
        print("WARNING: AI detection meta model not loaded. Using simple ensemble.")
        # Fallback to simple majority voting
        ai_votes = 0
        confidences = []
        
        for model, tokenizer in zip(models, tokenizers):
            model.eval()
            if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token 

            tokens = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                padding="max_length",
                truncation=True
            ).to(device)

            with torch.no_grad():
                logits = model(tokens["input_ids"], tokens["attention_mask"])

            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            pred = int(np.argmax(probs))
            conf = float(np.max(probs))
            
            if pred == 1:  # AI prediction
                ai_votes += 1
            confidences.append(conf)
        
        # Simple majority vote
        final_pred = 1 if ai_votes >= len(models) / 2 else 0
        conf = max(confidences)
        
        return final_pred, conf
    
    # Get predictions from all models
    preds = []
    confidences = []
    
    for model, tokenizer in zip(models, tokenizers):
        model.eval()
        if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token 

        tokens = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True
        ).to(device)

        with torch.no_grad():
            logits = model(tokens["input_ids"], tokens["attention_mask"])

        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred = int(np.argmax(probs))          
        conf = float(np.max(probs))           

        preds.append(pred)
        confidences.append(conf)
    
    # Fixed feature names with commas
    feature_names = ['roberta_pred', 'roberta_conf', 'gpt_pred', 'gpt_conf']
    
    if len(preds) + len(confidences) != len(feature_names):
        # Fallback if feature count doesn't match
        print("WARNING: Feature count mismatch. Using simple ensemble.")
        similar_votes = sum(1 for pred in preds if pred == 1)
        total_votes = len(preds)
        final_pred = 1 if similar_votes >= total_votes / 2 else 0
        conf = max(confidences)
        return final_pred, conf
    
    # Create DataFrame with correct feature names
    model_outputs = preds + confidences
    model_outputs_df = pd.DataFrame([model_outputs], columns=feature_names)
    
    try:
        # Get prediction from logistic model
        conf = np.max(ai_meta_model.predict_proba(model_outputs_df)) 
        final_pred = ai_meta_model.predict(model_outputs_df)

        if conf <= threshold and final_pred == 0:
            final_pred[0] = 1
            conf = 1 - conf
        return final_pred[0], conf
    except Exception as e:
        print(f"Error using logistic model: {e}")
        # Fallback to simple ensemble
        similar_votes = sum(1 for pred in preds if pred == 1)
        total_votes = len(preds)
        final_pred = 1 if similar_votes >= total_votes / 2 else 0
        conf = max(confidences)
        return final_pred, conf

def predict_logistic(models, meta_model, text, tokenizers, max_length=256, device=None, threshold=0.7):
    """
    Legacy function for backward compatibility - uses AI detection meta model
    """
    return predict_ai_detection_logistic(models, ai_detection_meta_model or meta_model, text, tokenizers, max_length, device, threshold)

def compute_similarity_logistic(text1, text2, similarity_meta_model, max_length=512):
    """
    Compute similarity using separate logistic model for similarity ensemble
    """
    try:
        print(f"DEBUG: Computing similarity with logistic model - Text1: {text1}, Text2: {text2}")
        
        if len(similarity_models) == 0:
            print("WARNING: No similarity models loaded. Using fallback method.")
            # Simple word overlap percentage
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if len(words1) == 0 and len(words2) == 0:
                return 100.0
            elif len(words1) == 0 or len(words2) == 0:
                return 0.0
            overlap = len(words1.intersection(words2))
            total = len(words1.union(words2))
            similarity_score = (overlap / total) * 100
            print(f"DEBUG: Fallback Similarity Score = {similarity_score:.2f}%")
            return similarity_score
        
        predictions = []
        confidences = []
        
        combined_text = text1 + " [SEP] " + text2
        
        for model, tokenizer in zip(similarity_models, similarity_tokenizers):
            try:
                tokens = tokenizer(
                    combined_text,
                    return_tensors="pt",
                    max_length=max_length,
                    padding="max_length",
                    truncation=True
                ).to(device)
                
                with torch.no_grad():
                    logits = model(tokens["input_ids"], tokens["attention_mask"])
                    probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
                    pred = int(np.argmax(probs))
                    conf = float(np.max(probs))
                    
                    predictions.append(pred)
                    confidences.append(conf)
                    
                    print(f"DEBUG: Similarity model prediction: {pred}, confidence: {conf:.4f}")
                    
            except Exception as model_error:
                print(f"Error with similarity model: {model_error}")
                continue
        
        if not predictions:
            print("All similarity models failed, using fallback")
            # Fallback to simple word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            overlap = len(words1.intersection(words2))
            total = len(words1.union(words2))
            return (overlap / total) * 100 if total > 0 else 0
        
        # Use logistic meta-model if available
        if similarity_meta_model is not None:
            # Prepare features for similarity logistic model
            model_outputs = predictions + confidences
            
            # Create feature names based on number of similarity models with commas
            feature_names = []
            for i, _ in enumerate(similarity_models):
                model_name = f"SimilarityModel_{i}"
                feature_names.extend([f"{model_name}_pred", f"{model_name}_conf"])
            
            # Adjust feature names if we have exactly 2 models (common case)
            if len(similarity_models) == 2:
                feature_names = ['pred_zero', 'conf_zero', 'pred_cross', 'conf_cross']
            
            # Ensure we have the right number of features
            if len(model_outputs) != len(feature_names):
                print(f"WARNING: Feature count mismatch. Expected {len(feature_names)}, got {len(model_outputs)}")
                # Use ensemble fallback
            else:
                model_outputs_df = pd.DataFrame([model_outputs], columns=feature_names)
                
                try:
                    similarity_probs = similarity_meta_model.predict_proba(model_outputs_df)
                    similarity_score = similarity_probs[0][1] * 100 
                    
                    print(f"DEBUG: Logistic Similarity Score = {similarity_score:.2f}%")
                    return max(0, min(similarity_score, 100))

                    
                except Exception as logistic_error:
                    print(f"Error using similarity logistic model: {logistic_error}")
        
        print("Using ensemble fallback for similarity")
        similar_votes = sum(1 for pred in predictions if pred == 1)
        total_votes = len(predictions)
        
        if similar_votes > total_votes / 2:
            # Majority says similar - use weighted average of confidences for similar predictions
            similar_confs = [conf for pred, conf in zip(predictions, confidences) if pred == 1]
            avg_confidence = np.mean(similar_confs)
            similarity_score = avg_confidence * 100
        else:
            # Majority says not similar - use inverse of weighted average
            dissimilar_confs = [conf for pred, conf in zip(predictions, confidences) if pred == 0]
            avg_confidence = np.mean(dissimilar_confs)
            similarity_score = (1 - avg_confidence) * 100
        
        # Clamp the score between 0 and 100
        similarity_score = max(0, min(similarity_score, 100))
        
        print(f"DEBUG: Ensemble Similarity Score = {similarity_score:.2f}%")
        print(f"DEBUG: Individual predictions: {predictions}")
        print(f"DEBUG: Individual confidences: {confidences}")
        
        return similarity_score
        
    except Exception as e:
        print(f"Error in compute_similarity_logistic function: {e}")
        # Return a safe fallback similarity score
        return 50.0  # Neutral similarity score

def compute_similarity_ensemble(text1, text2, max_length=512):
    """
    Enhanced similarity computation using logistic model if available, 
    fallback to ensemble method
    """
    return compute_similarity_logistic(text1, text2, similarity_meta_model, max_length)

def compute_similarity(text1, text2):
    """
    Backward compatibility wrapper for the original compute_similarity function
    """
    return compute_similarity_ensemble(text1, text2)

def evaluate_answer(student_answer, correct_answer, question_text, ai_threshold=0.7, similarity_threshold=60):
    try:
        print(f"DEBUG: Evaluating answer - Student: {student_answer}, Correct: {correct_answer}, Question: {question_text}")   
        
        if not isinstance(student_answer, str) or not student_answer.strip():
            return "❌ Invalid student answer."
        if not isinstance(correct_answer, str) or not correct_answer.strip():
            return "❌ Invalid correct answer."
        
        try:
            ai_threshold = float(ai_threshold)
            similarity_threshold = float(similarity_threshold)
        except (ValueError, TypeError):
            return "❌ Invalid threshold values. Thresholds must be numbers."
        
        # Use AI detection logistic model - SAME FUNCTION AS MANUAL TESTING
        prediction, confidence = predict_logistic(
            models, ai_detection_meta_model, student_answer, tokenizers, 
            device=device, threshold=ai_threshold
        )
        
        # Fix the logic to be consistent with your manual testing
        # Based on your test: prediction is True for AI, False for Human
        is_ai_generated = bool(prediction)  # True means AI-generated, False means human
        
        print(f"DEBUG: AI Detection - prediction: {prediction}, confidence: {confidence}, is_ai_generated: {is_ai_generated}")
        print(f"DEBUG: Prediction interpretation - {'AI' if prediction else 'Human'}")  
        
        if is_ai_generated:
            return "❌ Answer detected as AI-generated."
        
        # Use similarity logistic model
        similarity_score = compute_similarity_logistic(student_answer, correct_answer, similarity_meta_model)
        print(f"DEBUG: Similarity Score = {similarity_score:.2f}%")  
        
        if similarity_score >= similarity_threshold:
            return f"✅ Answer is correct. Similarity: {similarity_score:.2f}%"
        else:
            return f"❌ Answer is incorrect. Similarity: {similarity_score:.2f}%"
    
    except Exception as e:
        print(f"Error in evaluate_answer function: {e}")
        return f"❌ An error occurred while evaluating the answer: {e}"

model = models[0]
tokenizer = tokenizers[0]

example_text = "supervised learning is a type of machine learning where a model is trained on a labeled dataset, meaning that each input data point is paired with the correct output"
pred, cert = predict_logistic(models, ai_detection_meta_model, example_text, tokenizers, device=device)
print(f"Prediction: {'AI' if pred else 'Human'}, Certainty: {cert:.2f}")

example_text = "cs is computer science"
pred, cert = predict_logistic(models, ai_detection_meta_model, example_text, tokenizers, device=device)
print(f"Prediction: {'AI' if pred else 'Human'}, Certainty: {cert:.2f}")

__all__ = ['evaluate_answer', 'predict_logistic', 'predict_ai_detection_logistic', 
           'compute_similarity', 'compute_similarity_ensemble', 'compute_similarity_logistic',
           'models', 'tokenizers', 'similarity_models', 'similarity_tokenizers', 
           'model', 'tokenizer', 'ai_detection_meta_model', 'similarity_meta_model']