import os
from dotenv import load_dotenv
from paths import TOKEN_FILE_PATH

# Load .env variables
load_dotenv()

class ConfigManager:
    def __init__(self):
        # Load API configuration
        self.CLIENT_ID = os.getenv("FYERS_CLIENT_ID")
        self.SECRET_KEY = os.getenv("FYERS_SECRET_KEY")
        self.REDIRECT_URI = os.getenv("FYERS_REDIRECT_URI")
        self.RESPONSE_TYPE = "code"
        self.GRANT_TYPE = "authorization_code"
        # In-memory tokens (optional, for convenience)
        self._access_token = None
        self._refresh_token = None

    def set_tokens(self, access, refresh):
        self._access_token = access
        self._refresh_token = refresh

    def get_tokens(self):
        return self._access_token, self._refresh_token

    def save_tokens_to_file(self, access, refresh):
        with open(TOKEN_FILE_PATH, 'w') as f:
            f.write(f"Access Token: {access}\n")
            f.write(f"Refresh Token: {refresh}\n")

    def load_tokens_from_file(self):
        if not os.path.exists(TOKEN_FILE_PATH):
            return None
        with open(TOKEN_FILE_PATH, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:
                return None
            access = lines[0].split(":", 1)[-1].strip()
            refresh = lines[1].split(":", 1)[-1].strip()
            return access, refresh

    def ensure_tokens_loaded(self):
        """Utility: load tokens from file if not already set in memory."""
        if not self._access_token or not self._refresh_token:
            tokens = self.load_tokens_from_file()
            if tokens:
                self.set_tokens(*tokens)

# Create a global config manager instance for import convenience
config = ConfigManager()