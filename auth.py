import webbrowser
import logging
from fyers_apiv3 import fyersModel
from config import config

logging.basicConfig(filename="fyers_auth_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

session = fyersModel.SessionModel(
    client_id=config.CLIENT_ID,
    secret_key=config.SECRET_KEY,
    redirect_uri=config.REDIRECT_URI,
    response_type=config.RESPONSE_TYPE,
    grant_type=config.GRANT_TYPE
)

def authenticate():
    url = session.generate_authcode()
    webbrowser.open(url)
    logging.info("Opened Fyers login URL in browser.")

def generate_token(auth_code: str):
    if not auth_code:
        logging.warning("No auth code entered.")
        return False

    session.set_token(auth_code)
    response = session.generate_token()
    logging.info("Attempted to generate token.")

    if response.get("s") == "ok":
        access = response["access_token"]
        refresh = response["refresh_token"]
        config.set_tokens(access, refresh)
        config.save_tokens_to_file(access, refresh)
        logging.info("Tokens saved successfully.")
        return True
    else:
        logging.error(f"Token generation failed: {response.get('message')}")
        return False

def refresh_token():
    response = session.refresh_token()
    if response.get("s") == "ok":
        access = response["access_token"]
        refresh = response["refresh_token"]
        config.set_tokens(access, refresh)
        config.save_tokens_to_file(access, refresh)
        logging.info("Token refreshed successfully.")
        return True
    else:
        logging.error("Token refresh failed.")
        return False