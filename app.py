#!/usr/bin/env python3
"""
🦅 ZerAds ML CAPTCHA Solver API - Full Base64 Mode
Question: Base64, Choices: Base64
"""

import os
import pickle
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = os.environ.get("MODEL_PATH", "flower_model.pkl")
model = None
scaler = None

# ============================================================
# LOAD MODEL
# ============================================================
def load_model():
    global model, scaler
    try:
        logger.info(f"📚 Loading model from: {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            data = pickle.load(f)
            model = data['model']
            scaler = data['scaler']
        logger.info("✅ Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

# ============================================================
# FEATURE EXTRACTION
# ============================================================
def extract_features(img):
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
        
        if des is not None and len(des) > 0:
            sift_mean = des.mean(axis=0)
            sift_std = des.std(axis=0)
        else:
            sift_mean = np.zeros(128)
            sift_std = np.zeros(128)
        
        features = np.concatenate([h_hist, s_hist, v_hist, sift_mean, sift_std])
        return features
        
    except Exception as e:
        logger.error(f"❌ Feature extraction error: {e}")
        return None

# ============================================================
# PREDICT
# ============================================================
def predict_flower(img):
    try:
        features = extract_features(img)
        if features is None:
            return None, None
        
        features_scaled = scaler.transform([features])
        pred = model.predict(features_scaled)[0]
        confidence = model.predict_proba(features_scaled)[0][pred]
        
        return int(pred) + 1, float(confidence)
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return None, None

# ============================================================
# DECODE BASE64 IMAGE
# ============================================================
def decode_base64_image(b64_string):
    """Decode base64 string to OpenCV image"""
    try:
        # Remove data:image/jpeg;base64, prefix if present
        if ',' in b64_string:
            b64_string = b64_string.split(',')[1]
        
        img_bytes = base64.b64decode(b64_string)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error(f"❌ Failed to decode image")
            return None
            
        return img
        
    except Exception as e:
        logger.error(f"❌ Base64 decode error: {e}")
        return None

# ============================================================
# API ENDPOINT - /zerd (Full Base64)
# ============================================================
@app.route('/zerd', methods=['POST'])
def solve_captcha():
    """
    Solve flower CAPTCHA - Full Base64 Mode
    
    Request:
    {
        "question": "iVBORw0KGgoAAAANSUhEUgAA...",  # Base64
        "choices": [
            "iVBORw0KGgoAAAANSUhEUgAA...",  # Base64
            "iVBORw0KGgoAAAANSUhEUgAA...",  # Base64
            ...
        ]
    }
    
    Response:
    {
        "success": true,
        "answer": 3,
        "choice_url": "https://zerads.com/images/CaptchaPTC/3.jpg",
        "confidence": 87.3,
        "question_id": 3
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        # Get question (Base64)
        question_b64 = data.get('question')
        if not question_b64:
            return jsonify({
                "success": False,
                "error": "Question image required"
            }), 400
        
        # Get choices (List of Base64)
        choices_b64 = data.get('choices', [])
        if not choices_b64 or len(choices_b64) < 5:
            return jsonify({
                "success": False,
                "error": "At least 5 choices required"
            }), 400
        
        # Decode question
        logger.info("📥 Decoding question image...")
        question_img = decode_base64_image(question_b64)
        
        if question_img is None:
            return jsonify({
                "success": False,
                "error": "Failed to decode question image"
            }), 400
        
        logger.info("✅ Question decoded successfully")
        
        # Predict question
        question_id, question_conf = predict_flower(question_img)
        
        if question_id is None:
            return jsonify({
                "success": False,
                "error": "ML prediction failed"
            }), 500
        
        logger.info(f"📊 Question ID: {question_id} (Confidence: {question_conf*100:.1f}%)")
        
        # Process each choice
        choice_predictions = []
        
        for idx, choice_b64 in enumerate(choices_b64):
            logger.info(f"📥 Decoding choice {idx+1}...")
            
            choice_img = decode_base64_image(choice_b64)
            
            if choice_img is None:
                return jsonify({
                    "success": False,
                    "error": f"Failed to decode choice {idx+1}"
                }), 400
            
            pred_id, conf = predict_flower(choice_img)
            choice_predictions.append({
                "index": idx,
                "predicted_id": pred_id,
                "confidence": conf
            })
            
            logger.info(f"   Choice {idx+1}: ID {pred_id} (Confidence: {conf*100:.1f}%)")
        
        # Find matching choice
        answer_idx = None
        answer_conf = 0
        
        for pred in choice_predictions:
            if pred['predicted_id'] == question_id:
                if answer_idx is None or pred['confidence'] > answer_conf:
                    answer_idx = pred['index']
                    answer_conf = pred['confidence']
        
        # If no exact match, use closest ID
        if answer_idx is None:
            best_diff = 999
            for pred in choice_predictions:
                if pred['predicted_id'] is not None:
                    diff = abs(pred['predicted_id'] - question_id)
                    if diff < best_diff:
                        best_diff = diff
                        answer_idx = pred['index']
                        answer_conf = pred['confidence']
            
            if answer_idx is not None:
                logger.info(f"⚠️ No exact match, using closest: #{answer_idx+1}")
        
        if answer_idx is None:
            return jsonify({
                "success": False,
                "error": "No matching choice found"
            }), 500
        
        # Response (choice_url optional - client can use their own)
        response = {
            "success": True,
            "answer": answer_idx + 1,
            "choice_url": f"https://zerads.com/images/CaptchaPTC/{question_id}.jpg",
            "confidence": round(answer_conf * 100, 1),
            "question_id": question_id,
            "question_confidence": round(question_conf * 100, 1),
            "message": f"✅ Matched with choice #{answer_idx + 1}"
        }
        
        logger.info(f"✅ Answer: #{answer_idx + 1} → {response['choice_url']}")
        logger.info(f"   Confidence: {answer_conf*100:.1f}%")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================
@app.route('/', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "service": "ZerAds ML CAPTCHA Solver",
        "version": "2.0.0",
        "mode": "Full Base64"
    }), 200


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "model": "loaded" if model else "not loaded",
        "mode": "base64_only"
    }), 200


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    if not load_model():
        logger.error("❌ Exiting...")
        sys.exit(1)
    
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"🚀 Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port)
