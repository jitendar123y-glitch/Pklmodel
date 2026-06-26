#!/usr/bin/env python3
"""
🦅 ZerAds ML CAPTCHA Solver API - Render Deployment
Model: flower_model.pkl
"""

import os
import pickle
import base64
import numpy as np
import cv2
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from io import BytesIO
from PIL import Image
import sys
import logging

# Setup logging
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
    except FileNotFoundError:
        logger.error(f"❌ Model file not found: {MODEL_PATH}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

# ============================================================
# FEATURE EXTRACTION
# ============================================================
def extract_features(img):
    """Extract features from image for ML model"""
    try:
        # Resize to 64x64
        img = cv2.resize(img, (64, 64))
        
        # HSV histograms
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        v_hist = cv2.normalize(v_hist, v_hist).flatten()
        
        # SIFT features
        sift = cv2.SIFT_create()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        
        if des is not None and len(des) > 0:
            sift_mean = des.mean(axis=0)
            sift_std = des.std(axis=0)
        else:
            sift_mean = np.zeros(128)
            sift_std = np.zeros(128)
        
        # Combine all features
        features = np.concatenate([h_hist, s_hist, v_hist, sift_mean, sift_std])
        return features
        
    except Exception as e:
        logger.error(f"❌ Feature extraction error: {e}")
        return None

# ============================================================
# PREDICT FLOWER
# ============================================================
def predict_flower(img):
    """Predict flower ID from image"""
    try:
        features = extract_features(img)
        if features is None:
            return None, None
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Predict
        pred = model.predict(features_scaled)[0]
        confidence = model.predict_proba(features_scaled)[0][pred]
        
        return int(pred) + 1, float(confidence)
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return None, None

# ============================================================
# DOWNLOAD IMAGE FROM URL
# ============================================================
def download_image(url):
    """Download image from URL and return OpenCV image"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        }
        resp = requests.get(url, headers=headers, timeout=15)
        
        if resp.status_code == 200:
            nparr = np.frombuffer(resp.content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        logger.warning(f"Failed to download: {url} - Status: {resp.status_code}")
        return None
            
    except Exception as e:
        logger.error(f"Download error: {e}")
        return None

# ============================================================
# API ENDPOINT - /zerd
# ============================================================
@app.route('/zerd', methods=['POST'])
def solve_captcha():
    """
    Solve flower CAPTCHA
    
    Request:
    {
        "question": "base64_image_string",
        "choices": [
            "https://zerads.com/images/CaptchaPTC/1.jpg",
            "https://zerads.com/images/CaptchaPTC/2.jpg",
            ...
        ]
    }
    
    Response:
    {
        "success": true,
        "answer": 3,
        "choice_url": "https://zerads.com/images/CaptchaPTC/3.jpg",
        "confidence": 87.3
    }
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        # Get question image
        question_base64 = data.get('question')
        if not question_base64:
            return jsonify({
                "success": False,
                "error": "Question image required"
            }), 400
        
        # Get choices
        choices = data.get('choices', [])
        if not choices or len(choices) < 5:
            return jsonify({
                "success": False,
                "error": "At least 5 choices required"
            }), 400
        
        # Decode question image
        try:
            # Remove data:image/jpeg;base64, prefix if present
            if ',' in question_base64:
                question_base64 = question_base64.split(',')[1]
            
            img_bytes = base64.b64decode(question_base64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            question_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if question_img is None:
                return jsonify({
                    "success": False,
                    "error": "Failed to decode question image"
                }), 400
                
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Invalid image: {str(e)}"
            }), 400
        
        # Predict question flower ID
        logger.info("🤖 Predicting question...")
        question_id, question_conf = predict_flower(question_img)
        
        if question_id is None:
            return jsonify({
                "success": False,
                "error": "ML prediction failed"
            }), 500
        
        logger.info(f"📊 Question ID: {question_id} (Confidence: {question_conf*100:.1f}%)")
        
        # Download and predict each choice
        choice_predictions = []
        
        for idx, url in enumerate(choices):
            logger.info(f"📥 Downloading choice {idx+1}: {url}")
            img = download_image(url)
            
            if img is None:
                return jsonify({
                    "success": False,
                    "error": f"Failed to download choice {idx+1}"
                }), 400
            
            pred_id, conf = predict_flower(img)
            choice_predictions.append({
                "index": idx,
                "url": url,
                "predicted_id": pred_id,
                "confidence": conf
            })
            logger.info(f"   Choice {idx+1}: ID {pred_id} (Confidence: {conf*100:.1f}%)")
        
        # Find matching choice
        answer_idx = None
        answer_url = None
        answer_conf = 0
        
        for pred in choice_predictions:
            if pred['predicted_id'] == question_id:
                if answer_idx is None or pred['confidence'] > answer_conf:
                    answer_idx = pred['index']
                    answer_url = pred['url']
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
                        answer_url = pred['url']
                        answer_conf = pred['confidence']
            
            if answer_idx is not None:
                logger.info(f"⚠️ No exact match, using closest: #{answer_idx+1}")
        
        if answer_idx is None:
            return jsonify({
                "success": False,
                "error": "No matching choice found"
            }), 500
        
        # Response
        response = {
            "success": True,
            "answer": answer_idx + 1,
            "choice_url": answer_url,
            "confidence": round(answer_conf * 100, 1),
            "question_id": question_id,
            "question_confidence": round(question_conf * 100, 1),
            "message": f"✅ Matched with choice #{answer_idx + 1}"
        }
        
        logger.info(f"✅ Answer: #{answer_idx + 1} → {answer_url}")
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
        "version": "1.0.0"
    }), 200


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "model": "loaded" if model else "not loaded"
    }), 200


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Load model
    if not load_model():
        logger.error("❌ Exiting...")
        sys.exit(1)
    
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"🚀 Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port)
