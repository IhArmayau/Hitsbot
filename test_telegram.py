import requests

TELEGRAM_BOT_TOKEN = '8040574524:AAEBD8f-ksspzNfwv-MjnNjG4Y3tmapZsZA'
TELEGRAM_CHAT_ID = '995587401'

def send_telegram_message(message: str):
    url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message}
    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            print("✅ Message sent successfully!")
        else:
            print(f"❌ Failed: {response.text}")
    except Exception as e:
        print(f"⚠️ Error: {e}")

# --- TEST ---
send_telegram_message("🚀 Telegram bot connection successful!")
