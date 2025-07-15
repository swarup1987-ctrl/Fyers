from fyers_apiv3 import fyersModel
from config import config

class FyersService:
    def __init__(self):
        self.fyers = None
        self.ensure_session()

    def ensure_session(self):
        config.ensure_tokens_loaded()
        access, _ = config.get_tokens()
        if not access:
            self.fyers = None
        else:
            self.fyers = fyersModel.FyersModel(
                client_id=config.CLIENT_ID,
                is_async=False,
                token=access,
                log_path=""
            )

    def get_profile(self):
        self.ensure_session()
        if not self.fyers:
            raise Exception("Fyers session not initialized or token is missing.")
        return self.fyers.get_profile()

    def get_funds(self):
        self.ensure_session()
        if not self.fyers:
            raise Exception("Fyers session not initialized or token is missing.")
        return self.fyers.funds()