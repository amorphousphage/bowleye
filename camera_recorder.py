from threading import Thread, Event
from collections import deque
import av
import numpy as np
import time

class CameraRecorder(Thread):
    def __init__(self, settings, buffer, stop_event, is_pins_camera=False):
        self.settings = settings
        self.buffer = buffer
        self.stop_event = stop_event
        self.is_pins_camera = is_pins_camera
        self.running = True
        self.recording_active_event = Event()
        self.container = None
        self.frame_times = deque(maxlen=100)
        self.last_fps_report = time.time()

    def run(self):
        device = self.settings['pins_camera_path'] if self.is_pins_camera else self.settings['tracking_camera_path']
        width = self.settings['pins_camera_resolution'][0] if self.is_pins_camera else self.settings['tracking_camera_resolution'][0]
        height = self.settings['pins_camera_resolution'][1] if self.is_pins_camera else self.settings['tracking_camera_resolution'][1]
        fps = self.settings['fps_pins_camera'] if self.is_pins_camera else self.settings['fps_tracking_camera']

        try:
            self.container = av.open(
                device,
                format="v4l2",
                options={
                    "video_size": f"{width}x{height}",
                    "framerate": str(fps),
                    "pixel_format": "mjpeg"
                }
            )
        except (OSError, ValueError) as e:
            print("Camera not readable", f"{'Pins' if self.is_pins_camera else 'Tracking'} Camera could not be opened with PyAV: {e}")
            self.running = False
            return

        stream = self.container.streams.video[0]
        stream.thread_type = "AUTO"

        flushed = False

        while self.running:
            if not self.recording_active_event.is_set():
                flushed = False
                self.recording_active_event.wait()

            try:
                for frame in self.container.decode(stream):

                    if not self.running:
                        break

                    # Flush a few frames once after recording becomes active
                    if not flushed:
                        continue_flush = True
                        for _ in range(4):  # already consumed one frame here
                            try:
                                next(self.container.decode(stream))
                            except (StopIteration, av.AVError, OSError):
                                continue_flush = False
                                break
                        flushed = True
                        if continue_flush:
                            continue

                    img = frame.to_ndarray(format="bgr24")

                    if self.is_pins_camera and self.settings['pins_flipped'] == "Yes":
                        img = np.flip(img, axis=(0, 1))

                    # Copy the actual image to avoid glitches
                    self.buffer.set(img.copy())

                    break  # process exactly one frame per outer loop iteration

            except (StopIteration, av.AVError, OSError):
                continue

            # === FPS tracking ===
            now = time.time()
            self.frame_times.append(now)

            if now - self.last_fps_report >= 2.0:  # Report every 2 seconds
                if len(self.frame_times) >= 2:
                    elapsed = self.frame_times[-1] - self.frame_times[0]
                    avg_fps = len(self.frame_times) / elapsed if elapsed > 0 else 0
                    #print(f"[{self.__class__.__name__}] {'Pins' if self.is_pins_camera else 'Tracking'} FPS: {avg_fps:.2f}")
                self.last_fps_report = now

        self.stop_event.set()

    def stop(self):
        self.running = False
        if not self.recording_active_event.is_set():
            self.recording_active_event.set()
        if self.container:
            self.container.close()
