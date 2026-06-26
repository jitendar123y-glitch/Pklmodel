#!/usr/bin/env python3
"""
🦅 ZerAds ML CAPTCHA Solver API - Pre-download All Images
User sends: { "question": "base64", "choices": ["1.jpg","2.jpg","3.jpg","4.jpg","5.jpg"] }
Server uses pre-downloaded images
"""

import os
import pickle
import numpy as np
import cv2
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import logging
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = os.environ.get("MODEL_PATH", "flower_model.pkl")
IMAGE_BASE_URL = "https://zerads.com/images/CaptchaPTC"
IMAGE_CACHE = {}  # Store all images: { "1.jpg": img, "2.jpg": img, ... }
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
# DOWNLOAD ALL IMAGES (1-24)
# ============================================================
def download_all_images():
    """Download all 24 images from zerads.com"""
    global IMAGE_CACHE
    
    logger.info("📥 Downloading all 24 images from zerads.com...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    
    success_count = 0
    
    for i in range(1, 25):
        image_name = f"{i}.jpg"
        url = f"{IMAGE_BASE_URL}/{image_name}"
        
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            
            if resp.status_code == 200:
                content = resp.content
                
                # Check if valid image
                if len(content) < 500:
                    logger.warning(f"⚠️ {image_name} too small: {len(content)} bytes")
                    continue
                
                nparr = np.frombuffer(content, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None or img.size < 100:
                    logger.warning(f"⚠️ {image_name} invalid")
                    continue
                
                IMAGE_CACHE[image_name] = img
                success_count += 1
                logger.info(f"   ✅ {image_name} downloaded ({len(content)} bytes)")
            else:
                logger.warning(f"   ❌ {image_name} failed: {resp.status_code}")
                
        except Exception as e:
            logger.warning(f"   ❌ {image_name} error: {e}")
    
    logger.info(f"✅ Downloaded {success_count}/24 images")
    return success_count

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
        
        # Clean string
        b64_string = b64_string.strip().replace('\n', '').replace('\r', '')
        
        # Decode base64
        img_bytes = base64.b64decode(b64_string)
        
        if len(img_bytes) < 100:
            logger.error(f"❌ Image too small: {len(img_bytes)} bytes")
            return None
        
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None or img.size < 100:
            return None
            
        return img
        
    except Exception as e:
        logger.error(f"❌ Decode error: {e}")
        return None

# ============================================================
# GET IMAGE FROM CACHE
# ============================================================
def get_cached_image(image_name):
    """Get image from cache by name"""
    if image_name in IMAGE_CACHE:
        return IMAGE_CACHE[image_name]
    
    # Try with .jpg extension
    if not image_name.endswith('.jpg'):
        image_name = f"{image_name}.jpg"
    
    return IMAGE_CACHE.get(image_name)

# ============================================================
# API ENDPOINT - /zerd
# ============================================================
@app.route('/zerd', methods=['POST'])
def solve_captcha():
    """
    Solve flower CAPTCHA
    
    Request:
    {
        "question": "iVBORw0KGgoAAAANSUhEUgAA...",  # Base64
        "choices": ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg"]
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
                "error": "Question image (base64) required"
            }), 400
        
        # Get choices (image names)
        choices = data.get('choices', [])
        if not choices or len(choices) < 3:
            return jsonify({
                "success": False,
                "error": f"At least 3 choices required, got {len(choices)}"
            }), 400
        
        logger.info(f"📥 Question: Base64 ({len(question_b64)} chars)")
        logger.info(f"📥 Choices: {', '.join(choices)}")
        
        # Decode question image
        question_img = decode_base64_image(question_b64)
        
        if question_img is None:
            return jsonify({
                "success": False,
                "error": "Failed to decode question image"
            }), 400
        
        # Predict question
        question_id, question_conf = predict_flower(question_img)
        
        if question_id is None:
            return jsonify({
                "success": False,
                "error": "ML prediction failed for question"
            }), 500
        
        logger.info(f"📊 Question ID: {question_id} (Confidence: {question_conf*100:.1f}%)")
        
        # Process each choice from cache
        choice_predictions = []
        
        for idx, choice_name in enumerate(choices):
            # Ensure .jpg extension
            if not choice_name.endswith('.jpg'):
                choice_name = f"{choice_name}.jpg"
            
            logger.info(f"📥 Choice {idx+1}: {choice_name}")
            
            # Get from cache
            choice_img = get_cached_image(choice_name)
            
            if choice_img is None:
                logger.warning(f"⚠️ {choice_name} not in cache, skipping...")
                continue
            
            pred_id, conf = predict_flower(choice_img)
            
            if pred_id is not None:
                choice_predictions.append({
                    "index": idx,
                    "name": choice_name,
                    "predicted_id": pred_id,
                    "confidence": conf
                })
                logger.info(f"   Choice {idx+1}: ID {pred_id} (Confidence: {conf*100:.1f}%)")
            else:
                logger.warning(f"⚠️ Choice {choice_name} prediction failed")
        
        if len(choice_predictions) < 2:
            return jsonify({
                "success": False,
                "error": f"Only {len(choice_predictions)} valid choices, need at least 2"
            }), 500
        
        # Find matching choice
        answer_idx = None
        answer_conf = 0
        answer_name = None
        
        for pred in choice_predictions:
            if pred['predicted_id'] == question_id:
                if answer_idx is None or pred['confidence'] > answer_conf:
                    answer_idx = pred['index']
                    answer_conf = pred['confidence']
                    answer_name = pred['name']
        
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
                        answer_name = pred['name']
            
            if answer_idx is not None:
                logger.info(f"⚠️ No exact match, using closest: #{answer_idx+1} ({answer_name})")
        
        if answer_idx is None:
            return jsonify({
                "success": False,
                "error": "No matching choice found"
            }), 500
        
        # Build choice URL
        choice_url = f"{IMAGE_BASE_URL}/{answer_name}" if answer_name else f"{IMAGE_BASE_URL}/{question_id}.jpg"
        
        # Response
        response = {
            "success": True,
            "answer": answer_idx + 1,
            "choice_url": choice_url,
            "choice_name": answer_name,
            "confidence": round(answer_conf * 100, 1),
            "question_id": question_id,
            "question_confidence": round(question_conf * 100, 1),
            "message": f"✅ Matched with choice #{answer_idx + 1} ({answer_name})"
        }
        
        logger.info(f"✅ Answer: #{answer_idx + 1} ({answer_name})")
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
        "images_cached": len(IMAGE_CACHE),
        "service": "ZerAds ML CAPTCHA Solver",
        "version": "4.0.0",
        "mode": "Pre-downloaded Images"
    }), 200


@app.route('/cache', methods=['GET'])
def cache_status():
    return jsonify({
        "total_images": len(IMAGE_CACHE),
        "images": list(IMAGE_CACHE.keys())
    }), 200


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Load model
    if not load_model():
        logger.error("❌ Exiting...")
        sys.exit(1)
    
    # Download all images
    download_all_images()
    
    if len(IMAGE_CACHE) < 20:
        logger.warning(f"⚠️ Only {len(IMAGE_CACHE)} images downloaded, continuing anyway...")
    
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"🚀 Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port)
