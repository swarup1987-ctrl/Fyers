import logging
from db import init_db
from gui import launch_gui

# Logging setup
logging.basicConfig(filename="fyers_auth_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

if __name__ == "__main__":
    init_db()
    launch_gui()
