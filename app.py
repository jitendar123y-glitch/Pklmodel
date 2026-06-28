#!/usr/bin/env python3
"""
ZerAds ML API - Full Base64 Mode
All images (target + choices) sent as base64
Auto unzip flower_model.tar.gz if exists
"""

from flask import Flask, jsonify, request
import os
import pickle
import cv2
import numpy as np
import base64
import logging
import tarfile
import sys

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load ML model at startup
MODEL = None
SCALER = None
MODEL_FILE = 'flower_model.pkl'
MODEL_TAR = 'flower_model.tar.gz'

# ============================================================
# AUTO EXTRACT MODEL
# ============================================================

def extract_model():
    """Extract model from tar.gz if needed"""
    try:
        # Check if .pkl already exists
        if os.path.exists(MODEL_FILE):
            logger.info(f"✅ Model file already exists: {MODEL_FILE}")
            return True
        
        # Check if .tar.gz exists
        if os.path.exists(MODEL_TAR):
            logger.info(f"📦 Extracting {MODEL_TAR}...")
            
            with tarfile.open(MODEL_TAR, 'r:gz') as tar:
                # Extract all files
                tar.extractall()
                logger.info(f"✅ Extracted: {tar.getnames()}")
            
            # Check if .pkl extracted successfully
            if os.path.exists(MODEL_FILE):
                logger.info(f"✅ Model extracted successfully: {MODEL_FILE}")
                return True
            else:
                logger.error(f"❌ Model file not found after extraction: {MODEL_FILE}")
                return False
        else:
            logger.error(f"❌ Neither {MODEL_FILE} nor {MODEL_TAR} found!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Extraction failed: {e}")
        return False

# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    global MODEL, SCALER
    
    # First extract model if needed
    if not extract_model():
        logger.error("❌ Failed to extract model")
        return False
    
    try:
        logger.info("📚 Loading ML model...")
        with open(MODEL_FILE, 'rb') as f:
            data = pickle.load(f)
            MODEL = data['model']
            SCALER = data['scaler']
        logger.info("✅ Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False

# ============================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================

def base64_to_cv2(base64_string):
    """Convert base64 to OpenCV image"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Base64 decode error: {e}")
        return None

def recognize_flower(img):
    """ML model prediction"""
    if img is None or MODEL is None or SCALER is None:
        return None
    
    try:
        img = cv2.resize(img, (64, 64))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        v_hist = cv2.normalize(v_hist, v_hist).flatten()
        
        sift = cv2.SIFT_create()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        
        if des is not None:
            sift_mean = des.mean(axis=0)
            sift_std = des.std(axis=0)
        else:
            sift_mean = np.zeros(128)
            sift_std = np.zeros(128)
        
        features = np.concatenate([h_hist, s_hist, v_hist, sift_mean, sift_std])
        features = SCALER.transform([features])
        
        pred = MODEL.predict(features)[0]
        confidence = MODEL.predict_proba(features)[0][pred]
        
        return pred + 1, confidence
    except Exception as e:
        logger.error(f"Recognition error: {e}")
        return None

def get_visual_similarity(target_img, choice_img):
    """Calculate visual similarity between two images"""
    try:
        target_resized = cv2.resize(target_img, (64, 64))
        choice_resized = cv2.resize(choice_img, (64, 64))
        
        target_hsv = cv2.cvtColor(target_resized, cv2.COLOR_BGR2HSV)
        choice_hsv = cv2.cvtColor(choice_resized, cv2.COLOR_BGR2HSV)
        
        h_hist_target = cv2.calcHist([target_hsv], [0], None, [180], [0, 180])
        h_hist_choice = cv2.calcHist([choice_hsv], [0], None, [180], [0, 180])
        
        h_hist_target = cv2.normalize(h_hist_target, h_hist_target).flatten()
        h_hist_choice = cv2.normalize(h_hist_choice, h_hist_choice).flatten()
        
        hist_dist = cv2.compareHist(
            h_hist_target.reshape(-1, 1),
            h_hist_choice.reshape(-1, 1),
            cv2.HISTCMP_BHATTACHARYYA
        )
        similarity_score = 1.0 - hist_dist
        similarity_score = max(0, min(1, similarity_score))
        
        return float(similarity_score)
    except Exception as e:
        logger.error(f"Similarity calc error: {e}")
        return 0.0

# ============================================================
# ROUTES
# ============================================================

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': '🚀 ZerAds ML API',
        'mode': 'full_base64',
        'model': 'loaded' if MODEL else 'not loaded',
        'model_file': MODEL_FILE,
        'extracted': os.path.exists(MODEL_FILE),
        'endpoints': {
            'POST /predict': 'Predict flower from target + choices (all base64)',
            'GET /health': 'Health check'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'mode': 'full_base64',
        'model': 'loaded' if MODEL else 'not loaded',
        'model_exists': os.path.exists(MODEL_FILE)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict best choice for flower captcha
    ALL IMAGES AS BASE64 (target + choices)
    
    Request:
    {
        "target_image": "base64_encoded_image",
        "choices": [
            "base64_encoded_choice_1",
            "base64_encoded_choice_2",
            ...
        ]
    }
    
    Response:
    {
        "status": "success",
        "best_choice_index": 2,
        "predicted_flower_id": 5,
        "confidence": 0.987,
        "choice_scores": [0.45, 0.87, 0.23, ...]
    }
    """
    
    try:
        if MODEL is None or SCALER is None:
            return jsonify({'status': 'error', 'error': 'Model not loaded'}), 500
        
        data = request.json
        
        if not data:
            return jsonify({'status': 'error', 'error': 'No JSON data received'}), 400
        
        if 'target_image' not in data:
            return jsonify({'status': 'error', 'error': 'Missing target_image'}), 400
        
        if 'choices' not in data or not isinstance(data['choices'], list):
            return jsonify({'status': 'error', 'error': 'Missing choices (list of base64 images)'}), 400
        
        if len(data['choices']) == 0:
            return jsonify({'status': 'error', 'error': 'No choices provided'}), 400
        
        # Decode target image
        logger.info("Processing target image...")
        target_img = base64_to_cv2(data['target_image'])
        
        if target_img is None:
            return jsonify({'status': 'error', 'error': 'Failed to decode target_image'}), 400
        
        logger.info(f"Target image shape: {target_img.shape}")
        
        # ML Recognition on target
        result = recognize_flower(target_img)
        
        if result is None:
            return jsonify({'status': 'error', 'error': 'ML recognition failed'}), 400
        
        predicted_flower_id, confidence = result
        logger.info(f"Predicted flower: #{predicted_flower_id} ({confidence:.1%})")
        
        # Decode and analyze each choice (base64)
        logger.info(f"Processing {len(data['choices'])} choice images...")
        choice_scores = []
        
        for idx, choice_b64 in enumerate(data['choices']):
            choice_img = base64_to_cv2(choice_b64)
            
            if choice_img is None:
                logger.warning(f"Choice {idx}: Failed to decode")
                choice_scores.append(0.0)
                continue
            
            similarity = get_visual_similarity(target_img, choice_img)
            choice_scores.append(similarity)
            logger.info(f"Choice {idx}: similarity={similarity:.3f}")
        
        best_choice_idx = int(np.argmax(choice_scores))
        
        response = {
            'status': 'success',
            'best_choice_index': best_choice_idx,
            'predicted_flower_id': int(predicted_flower_id),
            'confidence': float(confidence),
            'choice_scores': choice_scores
        }
        
        logger.info(f"✅ Best choice: #{best_choice_idx}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in predict: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'error': str(e)}), 500

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    # Load model (auto extract if needed)
    if not load_model():
        logger.error("❌ Failed to load model. Exiting...")
        sys.exit(1)
    
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Server starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
