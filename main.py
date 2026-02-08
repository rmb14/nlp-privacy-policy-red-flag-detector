# SE Assignments

import json
import numpy as np
import re
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

# ------------ Config ------------
MODEL_DIR = "./privacy_bert_model"
BATCH_SIZE = 16
CONFIDENCE_THRESHOLD = 0.70
TOP_K = 3
DEVICE = "cpu"

app = Flask(__name__)
CORS(app)

# global vars for model
model = None
tokenizer = None
label_mappings = None
id2label = {}

def load_model():
    global model, tokenizer, label_mappings, id2label
    print(f"Loading model from {MODEL_DIR}...")
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    model = model.to(DEVICE) # since I use a mac i force it to use CPU cus it has issues with Apple Silicon Chips
    model.eval()
    
    mappings_file = os.path.join(MODEL_DIR, 'label_mappings.json')
    with open(mappings_file, 'r') as f:
        label_mappings = json.load(f)
        id2label = {int(k): v for k, v in label_mappings['id2label'].items()}
    
    print(f"Loaded {len(id2label)} labels")

def preprocess_text(text):
    if not text or not isinstance(text, str):
        return ""
    # Basic cleanup
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    if len(text.split()) < 3:
        return ""
    return text

def split_into_sentences(text):
    if not text:
        return []
    sentences = sent_tokenize(text)
    # clean sentences
    cleaned_sentences = []
    for sent in sentences:
        sent = preprocess_text(sent)
        if sent and len(sent) > 10:
            cleaned_sentences.append(sent)
    return cleaned_sentences

def boost_confidence_for_clear_patterns(sentence, label, confidence):
    # simple factoral boost for data sale/sharing label.
    if 'data sale or sharing for advertising' in label.lower():
        boosted_confidence = min(0.99, confidence * 1.4)  # 40% boost, cap at 99%
        return boosted_confidence
    return confidence

def format_label(label):
    return label.replace('_', ' ').replace(' / ', ' / ').title()

# TOOK HELP OF AI FOR THIS FUNCTION
def is_false_positive_enhanced(sentence, label, confidence):
    """Enhanced false positive detection for improved model"""
    sentence_lower = sentence.lower().strip()
    
    # Handle the NOT_CONCERNING label from improved model
    if label == 'NOT_CONCERNING':
        return True
    
    # Strong false positive indicators (enhanced patterns)
    patterns = [
        r'\bdo not\b.*\b(sell|share|collect|track|profile|disclose|engage|permit)\b',
        r'\bdoes not\b.*\b(sell|share|collect|track|profile|disclose|engage|permit)\b',
        r'\bwill not\b.*\b(sell|share|collect|track|profile|disclose|engage|permit)\b',
        r'\bnever\b.*\b(sell|share|collect|track|profile|disclose|engage|permit)\b',
        r'\bwe don\'t\b.*\b(sell|share|collect|track|profile|disclose|engage|permit)\b',
        r'\bnot permitted?\b.*\b(sell|share|collect|track|profile|disclose)\b',
        r'\bprohibited? from\b.*\b(sell|share|collect|track|profile|disclose)\b',
        r'\bdo not engage in\b.*\b(advertising|profiling|tracking)\b',
        r'\bdoes not cover\b',
        r'\bopt[- ]?out\b.*\b(of|from)\b',
        r'\byou may (change|update|edit|delete|control|manage)\b',
        r'\bview (their|your)\b.*\b(academic work|profile|information)\b',
        r'\bprovide.*\b(customer support|technical assistance|help)\b',
        r'\bassist.*\b(customers|users|you)\b.*\b(with|in)\b',
        r'\bprotect.*\b(your|user|customer)\b.*\b(privacy|data|information)\b'
    ]
    
    # Check against patterns
    for pattern in patterns:
        if re.search(pattern, sentence_lower):
            return True
    
    # Label-specific enhanced checks
    if 'data sale' in label.lower() or 'sharing' in label.lower():
        negative_indicators = [
            'do not sell', 'does not sell', 'will not sell', 'never sell',
            'do not rent', 'does not rent', 'will not rent', 'never rent',
            'do not share', 'does not share', 'will not share',
            'not permit', 'not authorized', 'prohibited from', 'not allowed',
            'opt out', 'opt-out', 'unsubscribe'
        ]
        if any(phrase in sentence_lower for phrase in negative_indicators):
            return True
    
    if 'extensive tracking' in label.lower() or 'profiling' in label.lower():
        control_indicators = [
            'you may', 'you can', 'you have the right', 'users can',
            'view your', 'edit your', 'change your', 'update your', 'delete your',
            'control your', 'manage your', 'access your'
        ]
        if any(phrase in sentence_lower for phrase in control_indicators):
            return True
    
    if 'data brokers' in label.lower():
        business_indicators = [
            'we provide', 'we offer', 'we help', 'we assist', 'we support',
            'our services', 'our platform', 'customer support', 'technical support'
        ]
        if any(phrase in sentence_lower for phrase in business_indicators):
            return True
    
    # Enhanced: Very short sentences are likely fragments
    if len(sentence.split()) < 8:
        return True
    
    # Enhanced: Legal compliance statements that aren't concerning
    compliance_patterns = [
        r'\bcomply with\b.*\b(law|regulation|legal)\b',
        r'\bas required by law\b',
        r'\blegal obligation\b',
        r'\bregulatory requirement\b'
    ]
    
    for pattern in compliance_patterns:
        if re.search(pattern, sentence_lower) and confidence < 0.90:
            return True
        
    return False

def batch_predict(sentences, threshold=CONFIDENCE_THRESHOLD):
    results = []
    
    for i in range(0, len(sentences), BATCH_SIZE):
        batch_sentences = sentences[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch_sentences, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        for sent, prob_vec in zip(batch_sentences, probs):
            top_idx = int(np.argmax(prob_vec))
            top_label = id2label.get(top_idx, f"label_{top_idx}")
            top_conf = float(prob_vec[top_idx])
            
            # confidence boosting
            boosted_conf = boost_confidence_for_clear_patterns(sent, top_label, top_conf)
            if (boosted_conf >= threshold and top_label != 'NOT_CONCERNING' and not is_false_positive_enhanced(sent, top_label, boosted_conf)):
                results.append({
                    "sentence": sent, 
                    "label": top_label,
                    "formatted_label": format_label(top_label),
                    "confidence": boosted_conf,
                    "original_confidence": top_conf
                })
    
    return results

@app.route('/classify', methods=['POST'])
def classify_text():
    data = request.get_json()
    text = data['text']
    threshold = data.get('threshold', CONFIDENCE_THRESHOLD)
    sentences = split_into_sentences(text)
    results = batch_predict(sentences, threshold)
    response = {
        "results": results,
        "summary": {
            "total_sentences": len(sentences),
            "concerning_sentences": len(results),
            "threshold_used": threshold
        }
    }
    
    return jsonify(response)

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=8000, debug=True)
