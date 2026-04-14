import os
import sys
import time
import configparser
import logging
from concurrent.futures import ThreadPoolExecutor
from coordinator import CoordinatorWorker
from logging.handlers import RotatingFileHandler

# Global logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

LANE_NUMBER = 12

def SafeRun(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        logging.exception("Unhandled exception in thread")
        raise

def load_settings(lane_number):
    config_path = f"settings_lane_{lane_number}.cfg"
    config = configparser.ConfigParser()
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No settings found for lane {lane_number} at {config_path}")
        else:
            config.read(config_path)
            return config
    except FileNotFoundError:
        logging.exception("Settings file missing")
        sys.exit(1)

def update_status(status):
    logging.info(f"[STATUS] {status}")

def main():
    logging.info("Loading settings...")
    settings = load_settings(LANE_NUMBER)

    logging.info("Starting CoordinatorWorker...")
    coordinator_executor = ThreadPoolExecutor(max_workers=1)
    recorder_worker = CoordinatorWorker(LANE_NUMBER, status_callback=update_status)

    # Submit to thread pool
    coordinator_executor.submit(SafeRun, recorder_worker.run)

    logging.info("Coordinator running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Stopping...")
        if hasattr(recorder_worker, 'stop'):
            recorder_worker.stop()
        coordinator_executor.shutdown(wait=True)
        logging.info("Stopped.")

if __name__ == "__main__":
    main()
