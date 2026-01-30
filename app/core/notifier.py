# import time
# import random
# import requests
# from .globals import CURRENT_CONFIG

# def send_whatsapp(cam_id, count, img_b64):
#     try:
#         base_url = CURRENT_CONFIG.get('waha_url').replace('/sendImage', '') # Ambil base URL
#         headers = {
#             "Content-Type": "application/json",
#             "X-Api-Key": CURRENT_CONFIG.get('api_key', '') 
#         }
        
#         payload_base = {
#             "session": CURRENT_CONFIG.get('session'),
#             "chatId": CURRENT_CONFIG.get('chat_id'),
#         }

#         # 1. Kirim "Seen" (Opsional, tapi bagus untuk reputasi)
#         requests.post(f"{base_url}/sendSeen", json=payload_base, headers=headers)

#         # 2. Mulai "Mengetik..."
#         requests.post(f"{base_url}/startTyping", json=payload_base, headers=headers)

#         # 3. Jeda Manusiawi (2 - 5 detik)
#         time.sleep(random.uniform(2.0, 5.0))

#         # 4. Stop "Mengetik..."
#         requests.post(f"{base_url}/stopTyping", json=payload_base, headers=headers)

#         # 5. Kirim Gambar
#         caption = (f"🚨 *ALERT CAM {cam_id}*\n"
#                    f"👥 Detect: *{count} People*\n"
#                    f"🕒 {time.strftime('%H:%M:%S')}")
        
#         payload_image = payload_base.copy()
#         payload_image.update({
#             "file": { 
#                 "mimetype": "image/jpeg", 
#                 "filename": "alert.jpg", 
#                 "data": img_b64 
#             },
#             "caption": caption
#         })
        
#         # Pastikan URL endpoint benar kembali ke /sendImage
#         requests.post(f"{base_url}/sendImage", json=payload_image, headers=headers, timeout=15)
#         print(f"✅ WA Sent (Human-Like) via Cam {cam_id}")

#     except Exception as e:
#         print(f"❌ WA Error: {e}")

# def send_telegram(cam_id, count, img_bytes):
#     try:
#         token = CURRENT_CONFIG.get('telegram_token')
#         chat_id = CURRENT_CONFIG.get('telegram_chat_id')
#         caption = (f"🚨 <b>ALERT CAM {cam_id}</b>\n"
#                    f"👥 Detect: <b>{count} People</b>\n"
#                    f"🕒 {time.strftime('%H:%M:%S')}")
#         url = f"https://api.telegram.org/bot{token}/sendPhoto"
#         files = {'photo': ('alert.jpg', img_bytes, 'image/jpeg')}
#         data = {'chat_id': chat_id, 'caption': caption, 'parse_mode': 'HTML'}
#         requests.post(url, data=data, files=files, timeout=10)
#     except Exception:
#         pass

#=================================================================

import time
import random
import requests
from .globals import CURRENT_CONFIG

def send_whatsapp(cam_id, caption, img_b64):
    """
    Mengirim pesan WhatsApp dengan gaya Human-Like.
    Menerima 'caption' (str) langsung dari Camera, bukan 'count'.
    """
    try:
        # Ambil base URL (misal: http://localhost:3000/api)
        waha_url = CURRENT_CONFIG.get('waha_url')
        if not waha_url: return

        base_url = waha_url.replace('/sendImage', '') 
        
        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": CURRENT_CONFIG.get('api_key', '') 
        }
        
        payload_base = {
            "session": CURRENT_CONFIG.get('session'),
            "chatId": CURRENT_CONFIG.get('chat_id'),
        }

        # 1. Kirim "Seen" (Agar terlihat seperti manusia membaca)
        requests.post(f"{base_url}/sendSeen", json=payload_base, headers=headers)

        # 2. Mulai "Mengetik..."
        requests.post(f"{base_url}/startTyping", json=payload_base, headers=headers)

        # 3. Jeda Manusiawi (Dikurangi sedikit agar notifikasi security lebih cepat)
        # Security alert sebaiknya cepat, jadi 1.5 - 3 detik cukup.
        time.sleep(random.uniform(1.5, 3.0)) 

        # 4. Stop "Mengetik..."
        requests.post(f"{base_url}/stopTyping", json=payload_base, headers=headers)

        # 5. Kirim Gambar & Caption
        # Kita gunakan langsung variabel 'caption' yang dikirim dari camera.py
        # Karena di camera.py sudah berisi: "🚨 DETEKSI... ID: Arya, Unknown"
        
        payload_image = payload_base.copy()
        payload_image.update({
            "file": { 
                "mimetype": "image/jpeg", 
                "filename": "alert.jpg", 
                "data": img_b64 
            },
            "caption": caption  # <--- INI PERUBAHAN UTAMANYA
        })
        
        # Kirim
        requests.post(f"{base_url}/sendImage", json=payload_image, headers=headers, timeout=15)
        print(f"✅ WA Sent (Human-Like) via Cam {cam_id}")

    except Exception as e:
        print(f"❌ WA Error: {e}")

def send_telegram(cam_id, caption, img_bytes):
    """
    Mengirim pesan Telegram.
    Menerima 'caption' (str) langsung dari Camera.
    """
    try:
        token = CURRENT_CONFIG.get('telegram_token')
        chat_id = CURRENT_CONFIG.get('telegram_chat_id')
        
        if not token or not chat_id: return
        
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        
        files = {
            'photo': ('alert.jpg', img_bytes, 'image/jpeg')
        }
        
        data = {
            'chat_id': chat_id,
            'caption': caption, # <--- Gunakan caption dari parameter
            'parse_mode': 'HTML'
        }
        
        requests.post(url, data=data, files=files, timeout=10)
        print(f"✅ Telegram Sent via Cam {cam_id}")
        
    except Exception as e:
        print(f"❌ Telegram Error: {e}")