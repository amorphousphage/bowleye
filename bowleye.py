import os
import sys
import time
import configparser
from concurrent.futures import ThreadPoolExecutor
from coordinator import CoordinatorWorker

LANE_NUMBER = 12

def load_settings(lane_number):
    config_path = f"settings_lane_{lane_number}.cfg"
    config = configparser.ConfigParser()
    if os.path.exists(config_path):
        config.read(config_path)
        return config
    else:
        print(f"[ERROR] No settings found for lane {lane_number} at {config_path}")
        sys.exit(1)

def update_status(status):
    print(f"[STATUS] {status}")

def main():
    print("[INFO] Loading settings...")
    settings = load_settings(LANE_NUMBER)

    print("[INFO] Starting CoordinatorWorker...")
    coordinator_executor = ThreadPoolExecutor(max_workers=1)
    recorder_worker = CoordinatorWorker(LANE_NUMBER, status_callback=update_status)

    # Submit to thread pool
    coordinator_executor.submit(recorder_worker.run)

    print("[INFO] Coordinator running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
        if hasattr(recorder_worker, 'stop'):
            recorder_worker.stop()
        coordinator_executor.shutdown(wait=True)
        print("[INFO] Stopped.")

if __name__ == "__main__":
    main()
