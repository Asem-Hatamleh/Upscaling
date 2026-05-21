import asyncio
import logging
import os
import queue
import ssl
import subprocess
import tempfile
import time
from asyncio import Task
from datetime import datetime
from queue import Queue
from threading import Thread

import cv2
import pytz
import requests
import websockets



class NodePlayer:
    BASE_URL = "https://dev-configmnvr.iotistic.com"

    def __init__(self, **kwargs):
       super().__init__()
       if not hasattr(self, 'logger') or self.logger is None:
          self.logger = logging.getLogger(self.__class__.__name__)
       self.username = kwargs.get('username', '')
       self.password = kwargs.get('password')

       self.imei = kwargs.get('imei')
       self.cam_id = kwargs.get('cam_id', 1)
       self.duration = kwargs.get('duration', 10)
       self.live_stream = kwargs.get('live_stream', False)
       self.show_video = kwargs.get('show_video', False)
       self.stream_type = kwargs.get('stream_type', 'sub')
       self.token = None
       self.start_time = kwargs.get('start_time', None)
       if isinstance(self.start_time, str):
          formats = [
             "%d/%m/%Y %H:%M:%S",
             "%m/%d/%Y, %I:%M:%S %p"
          ]

          dt = None
          for fmt in formats:
             try:
                dt = datetime.strptime(self.start_time, fmt)
                break
             except ValueError:
                continue

          if dt is None:
             raise ValueError(f"Time data '{self.start_time}' does not match any expected formats.")

          jordan_tz = pytz.timezone("Asia/Amman")
          if dt.tzinfo is None:
             dt = jordan_tz.localize(dt)

          self.start_time = int(dt.timestamp())
       self.logger.info("start_time   " + str(self.start_time))

       self.session_id = None
       self.stream_uri = None
       self.stop_event = asyncio.Event()
       self.output_file = None

       # For live display
       self.frame_queue = Queue(maxsize=300)
       self.display_thread = None

    # ------------------ Stream Type Detection ------------------

    def _is_hls_stream(self) -> bool:
       """Returns True if the stream URI is an HLS playlist (HTTP/HTTPS .m3u8)."""
       if self.stream_uri is None:
          return False
       uri_lower = self.stream_uri.lower()
       return uri_lower.startswith("http") and ".m3u8" in uri_lower

    # ------------------ API Helpers ------------------

    def get_token(self):
       url = f"{self.BASE_URL}/api/signin"
       resp = requests.post(url, json={"username": self.username, "password": self.password})
       resp.raise_for_status()
       self.token = resp.json().get("token")
       return self.token

    def check_connection(self) -> bool:
       url = f"{self.BASE_URL}/api/is-online/{self.imei}"
       headers = {"Authorization": f"Token {self.token}"}
       resp = requests.get(url, headers=headers)
       return resp.status_code == 200

    def start_playback(self, start_time: int):
       url = f"{self.BASE_URL}/stream/start-playback-stream/{self.imei}"
       headers = {"Authorization": f"Token {self.token}"}
       data = {"cam_id": self.cam_id, "starting_time": start_time, "duration": self.duration}
       resp = requests.post(url, headers=headers, json=data)
       resp.raise_for_status()
       data = resp.json()
       self.stream_uri = data.get("stream_uri")
       self.session_id = data.get("session_id")
       return data

    def start_live_stream(self):
       url = f"{self.BASE_URL}/stream/start-live-stream/{self.imei}"
       headers = {"Authorization": f"Token {self.token}"}
       data = {"cam_id": self.cam_id, "stream_type": self.stream_type}
       resp = requests.post(url, headers=headers, json=data)
       resp.raise_for_status()
       data = resp.json()
       self.stream_uri = data.get("stream_uri")
       self.session_id = data.get("session_id")
       return data

    def stop_stream(self, start_time: int):
       url = f"{self.BASE_URL}/stream/stop-stream/{self.session_id}"
       headers = {"Authorization": f"Token {self.token}"}
       data = {"cam_id": self.cam_id, "starting_time": start_time, "duration": self.duration}
       resp = requests.post(url, headers=headers, json=data)
       resp.raise_for_status()
       return resp.json()

    # ------------------ Video Display Thread ------------------

    def _display_loop(self, temp_file_path):
       """
       Separate thread that continuously reads and displays frames from the temp file.
       Used only for WebSocket/FLV streams where we write to a temp file incrementally.
       """
       self.logger.info("🎬 Starting display thread...")
       cap = None
       frame_count = 0

       waiting_logged = False
       while not self.stop_event.is_set():
          try:
             if cap is None:
                if not waiting_logged:
                   self.logger.info(f"cap is none, waiting for data (frames so far: {frame_count})")
                   waiting_logged = True
                if os.path.exists(temp_file_path) and os.path.getsize(temp_file_path) > 0:
                   cap = cv2.VideoCapture(temp_file_path)
                   if not cap.isOpened():
                      time.sleep(0.1)
                      continue
                   target_frame = max(frame_count - 1, 0)
                   cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                else:
                   time.sleep(0.1)
                   continue

             ret, frame = cap.read()

             if ret:
                frame_count += 1
                waiting_logged = False
                try:
                   self.frame_queue.put_nowait(frame)
                except queue.Full:
                   pass
                cv2.imshow("NodePlayer", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                   self.stop_event.set()
                   break
             else:
                time.sleep(0.03)
                if cap is not None:
                   cap.release()
                   cap = None

             time.sleep(0.01)

          except Exception as e:
             self.logger.info(f"⚠️ Display error: {e}")
             time.sleep(0.1)

       if cap is not None:
          cap.release()
       cv2.destroyAllWindows()
       self.logger.info(f"📊 Total frames displayed: {frame_count}")

    # ------------------ HLS Stream Handler ------------------

    async def _save_and_show_video_hls(self):
       """
       Handles HLS (.m3u8) streams.

       - Reads frames via cv2.VideoCapture (which uses libavformat/ffmpeg under the hood).
       - Simultaneously saves the stream to disk using an ffmpeg subprocess.
       - Pushes decoded frames into self.frame_queue for get_frame().
       """
       self.logger.info("📡 HLS stream detected: %s", self.stream_uri)

       # ---- 1. Start ffmpeg subprocess to save the stream to disk ----
       ffmpeg_proc = None
       try:
          ffmpeg_cmd = [
             "ffmpeg",
             "-y",  # overwrite output
             "-allowed_extensions", "ALL",
             "-protocol_whitelist", "file,http,https,tcp,tls,crypto",
             "-i", self.stream_uri,
             "-t", str(self.duration + 10),  # small buffer beyond requested duration
             "-c", "copy",
             self.output_file,
          ]
          self.logger.info("💾 Saving HLS stream via ffmpeg: %s", " ".join(ffmpeg_cmd))
          ffmpeg_proc = subprocess.Popen(
             ffmpeg_cmd,
             stdout=subprocess.DEVNULL,
             stderr=subprocess.PIPE,
          )
       except FileNotFoundError:
          self.logger.warning("⚠️  ffmpeg not found — stream will not be saved to disk.")

       # ---- 2. Read frames via OpenCV for the frame queue ----
       # Give the stream a moment to start buffering
       await asyncio.sleep(2)

       os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "protocol_whitelist;file,http,https,tcp,tls,crypto")

       cap = cv2.VideoCapture(self.stream_uri, cv2.CAP_FFMPEG)
       if not cap.isOpened():
          self.logger.error("❌ cv2.VideoCapture could not open HLS stream: %s", self.stream_uri)
          if ffmpeg_proc:
             ffmpeg_proc.terminate()
          return

       self.logger.info("✅ cv2.VideoCapture opened HLS stream")
       frame_count = 0
       start_ts = time.time()

       try:
          while not self.stop_event.is_set():
             ret, frame = cap.read()

             if not ret:
                # Transient read failure — wait and retry
                elapsed = time.time() - start_ts
                if elapsed > self.duration + 15:
                   self.logger.info("⏹️  HLS duration exceeded, stopping.")
                   break
                await asyncio.sleep(0.05)
                continue

             frame_count += 1
             try:
                self.frame_queue.put_nowait(frame)
             except queue.Full:
                pass  # Drop frame rather than block

             # Honour the requested duration
             if time.time() - start_ts >= self.duration + 10:
                self.logger.info("⏹️  Reached requested duration, stopping HLS reader.")
                break

             await asyncio.sleep(0)  # yield to event loop

       finally:
          cap.release()
          self.logger.info("📊 HLS frames captured: %d", frame_count)

          if ffmpeg_proc:
             ffmpeg_proc.terminate()
             try:
                _, stderr_out = ffmpeg_proc.communicate(timeout=10)
                if ffmpeg_proc.returncode not in (0, None, -15):
                   self.logger.warning("ffmpeg stderr: %s", stderr_out.decode(errors="replace")[-500:])
                else:
                   self.logger.info("✅ HLS video saved to %s", self.output_file)
             except subprocess.TimeoutExpired:
                ffmpeg_proc.kill()

    # ------------------ WebSocket / FLV Stream Handler ------------------

    async def _save_and_show_video_ws(self):
       """
       Handles WebSocket (ws:// / wss://) streams, typically FLV.
       Original implementation, unchanged in behaviour.
       """
       ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
       ssl_context.check_hostname = False
       ssl_context.verify_mode = ssl.CERT_NONE

       self.logger.info("📡 WebSocket stream detected: %s", self.stream_uri)

       temp_file = None
       if self.show_video:
          temp_file = tempfile.NamedTemporaryFile(suffix='.flv', delete=False)
          temp_file_path = temp_file.name
          self.logger.info(f"🎥 Temp file for display: {temp_file_path}")

          self.display_thread = Thread(target=self._display_loop, args=(temp_file_path,), daemon=True)
          self.display_thread.start()

       async with websockets.connect(self.stream_uri, ssl=ssl_context) as ws:
          self.logger.info("✅ Connected to WebSocket stream")
          self.logger.info(f"📹 Saving video to {self.output_file}")
          start_time = time.time()
          try:
             with open(self.output_file, 'wb') as raw_file:
                bytes_received = 0

                while not self.stop_event.is_set():
                   try:
                      message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                   except asyncio.TimeoutError:
                      self.logger.info("⏰ No data received for 5 seconds, stopping stream.")
                      break

                   if not isinstance(message, bytes):
                      self.logger.info("📩 Text frame from server: %s", message)
                      continue

                   raw_file.write(message)
                   bytes_received += len(message)

                   if self.show_video and temp_file:
                      temp_file.write(message)
                      temp_file.flush()

                   next_log_threshold = (bytes_received // (100 * 1024)) * (100 * 1024)
                   if next_log_threshold > 0 and bytes_received - len(message) < next_log_threshold:
                      self.logger.info(f"📥 Received: {bytes_received / 1024:.1f} KB")

                   if self.live_stream and (time.time() - start_time) > (self.duration + 50):
                      self.logger.info("Duration too long, stopping stream.")
                      break

          except Exception as e:
             self.logger.info("⚠️ Stream ended or error: %s", e)

          finally:
             if temp_file:
                temp_file.close()
                time.sleep(1)
                try:
                   os.unlink(temp_file_path)
                except Exception:
                   pass

             self.logger.info(f"✅ Video saved to {self.output_file}")
             self.logger.info(f"📦 Total bytes received: {bytes_received / 1024:.1f} KB")

    # ------------------ Unified dispatcher ------------------

    async def _save_and_show_video(self):
       """Routes to the correct handler based on the stream URI scheme."""
       if self._is_hls_stream():
          await self._save_and_show_video_hls()
       else:
          await self._save_and_show_video_ws()

    # ------------------ Main Runner ------------------

    async def run(self):
       self.logger.info("🔑 Getting token...")
       token = self.get_token()
       self.logger.info("Got token: %s", token)

       if not self.check_connection():
          self.logger.info("❌ Device is offline")
          return

       self.logger.info("✅ Device is online")
       if self.live_stream:
          playback_resp = self.start_live_stream()
       else:
          playback_resp = self.start_playback(self.start_time)
       self.logger.info("Playback response: %s", playback_resp)

       if not self.stream_uri:
          self.logger.info("❌ No stream URI received")
          return

       # Choose file extension based on stream type
       ext = "mp4" if self._is_hls_stream() else "flv"
       self.output_file = (
          f"{self.imei}_{self.cam_id}_{self.duration}"
          f"_{self.start_time or 0}_{int(time.time())}.{ext}"
       )
       self.logger.info(f"💾 Output file: {self.output_file}")

       await self._save_and_show_video()

       stop_resp = self.stop_stream(self.start_time)
       self.logger.info("🛑 Stopped stream: %s", stop_resp)

    async def get_frame(self):
       task: Task = asyncio.create_task(self.run())
       counter = 0
       while not task.done():
          try:
             frame = self.frame_queue.get(block=False)
          except queue.Empty:
             await asyncio.sleep(0)
             continue

          if counter % 5 == 0:
             yield {
                "image": frame,
                "name": self.output_file,
                "image_type": "bgr",
                "timestamp": time.time(),
             }
          counter += 1


# ------------------------
# Example usage
# ------------------------
async def main():
    downloader = NodePlayer(
       username="m.alawneh",
       password="sgQZ9Oou3CsP",
       imei="865847058020023",
       cam_id=4,
       duration=10,
       live_stream=True,
       start_time=1779269022,
       show_video=True
    )
    in_source_example = {
       "source_name": "node_player",
       "username": "",
       "password": "",
       "imei": "867105077527942",
       "cam_id": 1,
       "duration": 70,
       "live_stream": False,
       "start_time": 1778632048
    }
    await downloader.run()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info("Started")
    asyncio.run(main())
 