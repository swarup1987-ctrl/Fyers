# config.py

import os
from project_paths import data_path

# Fyers API Configuration
CLIENT_ID = "8CT51FCC5R-100"
SECRET_KEY = "CSFUO4A3IY"
REDIRECT_URI = "http://localhost:8000"
RESPONSE_TYPE = "code"
GRANT_TYPE = "authorization_code"

# Token Management
access_token = None
refresh_token = None

def set_tokens(access, refresh):
    global access_token, refresh_token
    access_token = access
    refresh_token = refresh

def get_tokens():
    return access_token, refresh_token

def save_tokens_to_file(access, refresh):
    filename = data_path("tokens.txt")
    with open(filename, 'w') as f:
        f.write(f"Access Token: {access}\n")
        f.write(f"Refresh Token: {refresh}\n")

def load_latest_token_from_file():
    filename = data_path("tokens.txt")
    if not os.path.exists(filename):
        return None

    with open(filename, 'r') as f:
        lines = f.readlines()
        if len(lines) < 2:
            return None
        access = lines[0].split(":", 1)[-1].strip()
        refresh = lines[1].split(":", 1)[-1].strip()
        return access, refresh
