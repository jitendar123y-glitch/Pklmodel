#!/usr/bin/env python3
"""
🔍 Render URL Checker - Open URL and Show Status
"""

import requests
import sys

# ============================================================
# URL TO CHECK
# ============================================================
URL = "https://zerads.com/images/CaptchaPTC/23.jpg"

def check_url(url):
    """Open URL and show status"""
    print("\n" + "="*60)
    print(f"🔍 CHECKING URL: {url}")
    print("="*60)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 Chrome/149.0.7827.91 Mobile Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }
    
    try:
        # GET request with timeout
        resp = requests.get(url, headers=headers, timeout=10)
        
        # Show status
        print(f"\n📊 STATUS: {resp.status_code}")
        print(f"📊 REASON: {resp.reason}")
        print(f"📊 CONTENT-TYPE: {resp.headers.get('Content-Type', 'Unknown')}")
        print(f"📊 CONTENT-LENGTH: {resp.headers.get('Content-Length', 'Unknown')} bytes")
        print(f"📊 SERVER: {resp.headers.get('Server', 'Unknown')}")
        
        # Check if image
        if resp.status_code == 200:
            print(f"\n✅ SUCCESS! Image loaded successfully!")
            
            # Save image to check
            with open("test_image.jpg", "wb") as f:
                f.write(resp.content)
            print(f"💾 Image saved: test_image.jpg ({len(resp.content)} bytes)")
            
        elif resp.status_code == 403:
            print(f"\n❌ BLOCKED! Status 403 - Forbidden")
            
        elif resp.status_code == 404:
            print(f"\n❌ NOT FOUND! Status 404")
            
        elif resp.status_code == 503:
            print(f"\n❌ BLOCKED! Status 503 - Service Unavailable")
            
        else:
            print(f"\n⚠️ UNKNOWN STATUS: {resp.status_code}")
            
        return resp
            
    except requests.exceptions.Timeout:
        print(f"\n❌ TIMEOUT! Server not responding")
        return None
        
    except requests.exceptions.ConnectionError:
        print(f"\n❌ CONNECTION ERROR! Cannot reach server")
        return None
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return None


# ============================================================
# CHECK MULTIPLE IMAGES
# ============================================================

def check_multiple_images():
    """Check all 24 images"""
    print("\n" + "="*60)
    print("🔍 CHECKING ALL 24 IMAGES")
    print("="*60)
    
    base_url = "https://zerads.com/images/CaptchaPTC"
    
    results = []
    
    for i in range(1, 25):
        url = f"{base_url}/{i}.jpg"
        
        try:
            resp = requests.get(url, timeout=10)
            status = resp.status_code
            results.append((i, status))
            
            if status == 200:
                print(f"✅ {i}.jpg - {status}")
            else:
                print(f"❌ {i}.jpg - {status}")
                
        except Exception as e:
            print(f"❌ {i}.jpg - ERROR: {e}")
            results.append((i, "ERROR"))
        
        time.sleep(0.5)  # Rate limit
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    success = sum(1 for _, s in results if s == 200)
    failed = sum(1 for _, s in results if s != 200)
    
    print(f"✅ Success: {success}/24")
    print(f"❌ Failed: {failed}/24")
    
    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import time
    
    # Check single URL
    check_url(URL)
    
    # Check all images
    print("\n" + "="*60)
    print("📸 CHECKING ALL IMAGES")
    print("="*60)
    check_multiple_images()
