import asyncio
import logging
import os
import queue
import re
import ssl
import subprocess
import tempfile
import time
from asyncio import Task
from datetime import datetime
from queue import Queue
from threading import Event, Thread

import cv2
import numpy as np
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
       self.debug_decode = kwargs.get('debug_decode', False)
       self.stream_type = kwargs.get('stream_type', 'sub')
       # Cap ffmpeg's decode rate to pipeline-sustainable fps so frames in
       # frame_queue stay consecutive instead of being evicted by drop-oldest.
       # 0 / None = decode every frame.
       self.decode_fps = kwargs.get('decode_fps') or 0
       self.queue_maxsize = int(kwargs.get('queue_maxsize') or 8)
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

       # For live display. Small queue keeps consumed frames recent; drop-oldest
       # policy in producer handles overflow.
       self.frame_queue = Queue(maxsize=self.queue_maxsize)
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
       Background thread that continuously decodes frames from the WebSocket/FLV
       temp file (which the WS receiver is appending to) and pushes them onto
       ``self.frame_queue``.

       Pure feeder: does NOT call cv2.imshow / cv2.waitKey here -- those are Qt
       calls that must run on the main thread. External consumers (pipeline_live,
       __main__ test) own display.

       Resilience:
         * Wait for an initial buffer (~256 KB) before opening cv2.VideoCapture
           so the FLV header is fully present.
         * Keep cap open across reads to avoid re-parsing the growing file.
         * On read failure, sleep + retry; only release+reopen after a long stall.
         * Drop-oldest policy on queue full to keep live latency low.
       """
       self.logger.info("🎬 Starting display (decode) thread...")
       cap = None
       frame_count = 0
       consecutive_fails = 0
       initial_buffer_bytes = 256 * 1024

       # 1) wait for enough bytes to make a valid FLV header before opening cap
       while not self.stop_event.is_set():
          try:
             if os.path.exists(temp_file_path) and os.path.getsize(temp_file_path) >= initial_buffer_bytes:
                break
          except OSError:
             pass
          time.sleep(0.05)

       # 2) decode loop -- cap stays open across reads.
       # CRITICAL: cv2.VideoCapture on a growing FLV restarts at frame 0 every reopen,
       # and cv2's seek on FLV (CAP_PROP_POS_FRAMES) is unreliable -- the demuxer
       # often ignores it or lands a few frames off. Result: the same real-world
       # event (e.g. driver reaching for phone) replays in the output.
       #
       # Fix: after reopen, use cap.grab() in a tight loop to advance the demuxer
       # past frame_count frames WITHOUT decoding (grab is much cheaper than read).
       # This guarantees we resume exactly at the next un-emitted frame.
       while not self.stop_event.is_set():
          try:
             if cap is None:
                cap = cv2.VideoCapture(temp_file_path)
                if not cap.isOpened():
                   cap = None
                   time.sleep(0.05)
                   continue
                # Catch up to frame_count by grabbing (no decode) so the demuxer
                # pointer lands at the next un-emitted frame.
                if frame_count > 0:
                   to_skip = frame_count
                   skipped = 0
                   t0 = time.time()
                   while skipped < to_skip:
                      if self.stop_event.is_set():
                         break
                      ok = cap.grab()
                      if not ok:
                         break
                      skipped += 1
                      # Yield occasionally if file hasn't flushed enough yet.
                      if skipped % 200 == 0:
                         time.sleep(0.005)
                   if self.debug_decode:
                      self.logger.info(f"[decode] reopen caught up: skipped={skipped}/{to_skip} in {(time.time()-t0)*1000:.0f}ms")

             ret, frame = cap.read()
             if ret:
                frame_count += 1
                consecutive_fails = 0
                if self.debug_decode and frame_count % 30 == 0:
                   try:
                      sz = os.path.getsize(temp_file_path)
                   except OSError:
                      sz = -1
                   pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                   self.logger.info(f"[decode] emit#{frame_count} pos={pos} temp_file={sz/1024:.0f}KB qsize={self.frame_queue.qsize()}")
                try:
                   self.frame_queue.put_nowait(frame)
                except queue.Full:
                   # drop oldest, keep latest -- live policy
                   try:
                      self.frame_queue.get_nowait()
                   except queue.Empty:
                      pass
                   try:
                      self.frame_queue.put_nowait(frame)
                   except queue.Full:
                      pass
             else:
                consecutive_fails += 1
                time.sleep(0.01)
                # tail of growing file: read() returns False until more bytes flushed.
                # After ~2s of nothing, the demuxer is probably wedged on a bad
                # packet boundary -- reopen + grab-catchup to resync.
                if consecutive_fails > 200:
                   cap.release()
                   cap = None
                   consecutive_fails = 0

          except Exception as e:
             self.logger.info(f"⚠️ Decode error: {e}")
             time.sleep(0.1)

       if cap is not None:
          cap.release()
       self.logger.info(f"📊 Total frames decoded: {frame_count}")

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
       Handles WebSocket (ws:// / wss://) FLV streams.

       Pipes WS bytes into an ffmpeg subprocess that:
         * tolerates the server's bogus PreviousTagSize fields
           (-fflags +discardcorrupt -err_detect ignore_err),
         * emits decoded BGR frames on stdout as raw video,
         * keeps a copy of the original FLV on disk.

       cv2.VideoCapture on a growing temp file (old path) had no way to ignore
       corrupt FLV tags, which caused "Packet mismatch" floods, decoder MB
       errors, silent frame drops, and resync stalls. ffmpeg with the right
       flags swallows that noise and gives us a clean frame stream.
       """
       ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
       ssl_context.check_hostname = False
       ssl_context.verify_mode = ssl.CERT_NONE

       self.logger.info("📡 WebSocket stream detected: %s", self.stream_uri)

       ffmpeg_cmd = [
          "ffmpeg", "-hide_banner", "-loglevel", "info",
          "-fflags", "+discardcorrupt+genpts",
          "-err_detect", "ignore_err",
          "-f", "flv", "-i", "pipe:0",
          "-map", "0:v:0", "-an",
       ]
       if self.decode_fps and self.decode_fps > 0:
          # -vsync cfr + -r ensures evenly-spaced frame drops at source rate
          # rather than chunky drop-oldest evictions in frame_queue.
          ffmpeg_cmd += ["-vsync", "cfr", "-r", str(self.decode_fps)]
       ffmpeg_cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
       try:
          proc = subprocess.Popen(
             ffmpeg_cmd,
             stdin=subprocess.PIPE,
             stdout=subprocess.PIPE,
             stderr=subprocess.PIPE,
             bufsize=0,
          )
       except FileNotFoundError:
          self.logger.error("ffmpeg not found in PATH; cannot decode WS stream")
          return

       size_event = Event()
       size_box = {"w": 0, "h": 0}
       frames_decoded = {"n": 0}

       def _stderr_reader():
          pat = re.compile(r"Video:.*?\b(\d{2,5})x(\d{2,5})\b")
          assert proc.stderr is not None
          for raw in iter(proc.stderr.readline, b""):
             try:
                line = raw.decode(errors="replace").rstrip()
             except Exception:
                continue
             if not size_event.is_set():
                m = pat.search(line)
                if m:
                   size_box["w"] = int(m.group(1))
                   size_box["h"] = int(m.group(2))
                   size_event.set()
                   self.logger.info(f"🎬 ffmpeg decode: {size_box['w']}x{size_box['h']}")
             # Suppress known-noisy lines, surface real fatals only.
             if "Packet mismatch" in line:
                continue
             if "error while decoding" in line:
                continue
             if line and ("error" in line.lower() or "fail" in line.lower()):
                self.logger.debug("[ffmpeg] %s", line)

       def _stdout_reader():
          if not size_event.wait(timeout=20.0):
             self.logger.error("ffmpeg never reported video size; aborting decoder")
             return
          w, h = size_box["w"], size_box["h"]
          frame_bytes = w * h * 3
          assert proc.stdout is not None
          buf = bytearray()
          while not self.stop_event.is_set():
             need = frame_bytes - len(buf)
             chunk = proc.stdout.read(need)
             if not chunk:
                break
             buf += chunk
             if len(buf) < frame_bytes:
                continue
             frame = np.frombuffer(bytes(buf), dtype=np.uint8).reshape(h, w, 3).copy()
             buf.clear()
             frames_decoded["n"] += 1
             try:
                self.frame_queue.put_nowait(frame)
             except queue.Full:
                try:
                   self.frame_queue.get_nowait()
                except queue.Empty:
                   pass
                try:
                   self.frame_queue.put_nowait(frame)
                except queue.Full:
                   pass

       stderr_thread = Thread(target=_stderr_reader, daemon=True)
       stdout_thread = Thread(target=_stdout_reader, daemon=True)
       stderr_thread.start()
       stdout_thread.start()

       bytes_received = 0
       try:
          async with websockets.connect(self.stream_uri, ssl=ssl_context) as ws:
             self.logger.info("✅ Connected to WebSocket stream")
             self.logger.info(f"📹 Saving video to {self.output_file}")
             start_time = time.time()
             try:
                with open(self.output_file, 'wb') as raw_file:
                   while not self.stop_event.is_set():
                      try:
                         message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                      except asyncio.TimeoutError:
                         self.logger.info("⏰ No data received for 5 seconds, stopping stream.")
                         break

                      if not isinstance(message, bytes):
                         self.logger.info("📩 Text frame from server: %s", message)
                         continue

                      bytes_received += len(message)
                      raw_file.write(message)
                      try:
                         assert proc.stdin is not None
                         proc.stdin.write(message)
                      except (BrokenPipeError, OSError) as e:
                         self.logger.warning(f"ffmpeg stdin closed: {e}")
                         break

                      next_log_threshold = (bytes_received // (100 * 1024)) * (100 * 1024)
                      if next_log_threshold > 0 and bytes_received - len(message) < next_log_threshold:
                         self.logger.info(f"📥 Received: {bytes_received / 1024:.1f} KB")

                      if self.live_stream and (time.time() - start_time) > (self.duration + 50):
                         self.logger.info("Duration too long, stopping stream.")
                         break
             except Exception as e:
                self.logger.info("⚠️ Stream ended or error: %s", e)
       finally:
          try:
             if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
          except Exception:
             pass
          try:
             proc.wait(timeout=5)
          except subprocess.TimeoutExpired:
             proc.kill()
          stdout_thread.join(timeout=2)
          stderr_thread.join(timeout=2)
          self.logger.info(f"✅ Video saved to {self.output_file}")
          self.logger.info(f"📦 Total bytes received: {bytes_received / 1024:.1f} KB")
          self.logger.info(f"📊 Total frames decoded: {frames_decoded['n']}")

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
       out_dir = "recordings"
       os.makedirs(out_dir, exist_ok=True)
       self.output_file = os.path.join(
          out_dir,
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
 