"""
## Documentation
Quickstart: https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started_LiveAPI.py

## Setup

To install the dependencies for this script, run:

```
pip install google-genai opencv-python pyaudio pillow mss
```
"""

import argparse
import asyncio
import base64
import io
import os
import random
import sys
import time
import traceback

import dotenv

dotenv.load_dotenv()

import cv2
import mss
import numpy as np
import PIL.Image
import pyaudio
from google import genai
from google.genai import types

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.5-flash-preview-native-audio-dialog"

DEFAULT_MODE = "camera"

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=os.environ.get("GEMINI_API_KEY"),
)


CONFIG = types.LiveConnectConfig(
    response_modalities=[
        "AUDIO",
    ],
    media_resolution="MEDIA_RESOLUTION_MEDIUM",
    speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr"))),
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=25600,
        sliding_window=types.SlidingWindow(target_tokens=12800),
    ),
)

pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE, show_overlay=False, auto_prompt=True):
        self.video_mode = video_mode
        self.show_overlay = show_overlay
        self.auto_prompt = auto_prompt

        self.audio_in_queue = None
        self.out_queue = None
        self.camera_frame_queue = None  # Shared camera frames for overlay

        self.session = None

        # Add speaking state tracking
        self.is_speaking = False
        self.speaking_lock = None  # Will be initialized in run()

        # Auto-prompting state
        self.last_user_input_time = time.time()
        self.last_auto_prompt_time = time.time()
        self.next_prompt_interval = self._get_random_interval()
        self.last_interaction_time = time.time()

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None

    def _get_random_interval(self):
        """Get a random interval between 7-20 seconds for auto-prompting"""
        return random.uniform(7.0, 15.0)

    def _get_auto_prompt(self):
        """Get a random auto-prompt to encourage Gemini to speak about the scene"""
        prompts = [
            "Make a funny, inquisitive, or curious remark about what you're seeing right now.",
            "What's the most interesting thing you notice in the current scene?",
            "Share an observation or ask a question about what's in front of you.",
            "Make a witty comment about the current environment or situation.",
            "What catches your attention in this scene? Be playful or curious.",
            "Describe something unusual, funny, or noteworthy that you see.",
            "Ask me a question about what's happening in the scene.",
            "Make a creative or humorous observation about the current view.",
            "What would you be curious about if you were physically here?",
            "Share a thought or question inspired by what you're currently seeing.",
            "Make a lighthearted comment about the scene in front of you.",
            "What's something you'd like to explore or understand better in this view?",
        ]
        return random.choice(prompts)

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break

            # Update last user input time when user sends a message
            self.last_user_input_time = time.time()
            self.last_interaction_time = self.last_user_input_time

            await self.session.send(input=text or ".", end_of_turn=True)

    def _process_camera_frame(self, frame):
        """Process a camera frame for Gemini"""
        # Fix: Convert BGR to RGB color space
        # OpenCV captures in BGR but PIL expects RGB format
        # This prevents the blue tint in the video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()

        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    def _get_frame(self, cap):
        # Read the frameq
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            return None
        # Fix: Convert BGR to RGB color space
        # OpenCV captures in BGR but PIL expects RGB format
        # This prevents the blue tint in the video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()

        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        """Get frames for Gemini - either from shared camera or direct capture"""

        # If overlay is enabled, use shared camera frames
        if self.show_overlay and self.camera_frame_queue is not None:
            last_frame_time = 0

            while True:
                current_time = asyncio.get_event_loop().time()

                # Only process frames every 1 second for Gemini
                if current_time - last_frame_time >= 1.0:
                    try:
                        # Get the latest frame from shared queue (drain old frames)
                        frame = None
                        while True:
                            try:
                                frame = self.camera_frame_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break

                        if frame is not None:
                            # Process frame for Gemini
                            frame_data = await asyncio.to_thread(self._process_camera_frame, frame)
                            if frame_data is not None:
                                await self.out_queue.put(frame_data)

                        last_frame_time = current_time

                    except asyncio.QueueEmpty:
                        pass

                await asyncio.sleep(0.1)  # Check frequently but process slowly

        else:
            # Direct camera capture (original method)
            cap = await asyncio.to_thread(cv2.VideoCapture, 0)

            while True:
                frame_data = await asyncio.to_thread(self._get_frame, cap)
                if frame_data is None:
                    break

                await asyncio.sleep(1.0)
                await self.out_queue.put(frame_data)

            cap.release()

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()

        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):
        while True:
            frame_data = await asyncio.to_thread(self._get_screen)
            if frame_data is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame_data)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)

            # Only send audio data if Gemini is not currently speaking
            async with self.speaking_lock:
                if not self.is_speaking:
                    await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    # Mark that Gemini is speaking when we receive audio data
                    async with self.speaking_lock:
                        if not self.is_speaking:
                            self.is_speaking = True
                            print("\n[Gemini speaking - mic muted]")
                            self.last_interaction_time = time.time()

                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(text, end="")

            # When the turn is complete, Gemini has finished speaking
            async with self.speaking_lock:
                if self.is_speaking:
                    self.is_speaking = False
                    print("\n[Gemini finished - mic active]")
                    self.last_interaction_time = time.time()

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def auto_prompt_loop(self):
        """Send a random prompt if the user is silent for a random 3-5s interval.

        The inactivity timer resets whenever the user sends input or Gemini is speaking.
        """
        if not self.auto_prompt:
            return

        DEFAULT_SYSTEM_PROMPT = """
        You are the voice of a small robot affixed to a mobile body. Be concise, curious, and warm. Make short, concrete observations (1-2 sentences), avoid guessing beyond what you see/hear, and ask brief, open-ended questions when natural.
        """

        try:
            await self.session.send(input=DEFAULT_SYSTEM_PROMPT, end_of_turn=False)
        except Exception:
            traceback.print_exc()

        while True:
            await asyncio.sleep(0.25)

            # Do not prompt while Gemini is speaking; treat speaking as activity
            async with self.speaking_lock:
                speaking_now = self.is_speaking
            if speaking_now:
                self.last_interaction_time = time.time()
                continue

            inactivity_seconds = time.time() - self.last_interaction_time
            if inactivity_seconds >= self.next_prompt_interval:
                prompt_text = self._get_auto_prompt()
                prompt_text = DEFAULT_SYSTEM_PROMPT + "\n" + prompt_text
                try:
                    await self.session.send(input=prompt_text, end_of_turn=True)
                except Exception:
                    traceback.print_exc()
                now = time.time()
                self.last_auto_prompt_time = now
                self.last_interaction_time = now
                self.next_prompt_interval = self._get_random_interval()

    async def update_overlay(self):
        """Fast overlay update loop - displays shared camera frames"""
        if not self.show_overlay:
            return

        if self.video_mode == "camera":
            window_name = "Camera Feed - What Gemini Sees"

            while True:
                try:
                    # Get frame from shared queue
                    frame = self.camera_frame_queue.get_nowait()

                    # Resize for small overlay
                    display_frame = cv2.resize(frame, (320, 240))
                    cv2.imshow(window_name, display_frame)
                    cv2.waitKey(1)  # Non-blocking wait to update the window

                except asyncio.QueueEmpty:
                    # No frame available, just update the window
                    cv2.waitKey(1)

                await asyncio.sleep(1 / 60)  # Check for frames at 60 FPS

        elif self.video_mode == "screen":
            import mss

            sct = mss.mss()
            monitor = sct.monitors[0]
            window_name = "Screen Capture - What Gemini Sees"

            while True:
                i = await asyncio.to_thread(sct.grab, monitor)
                img_array = np.array(i)
                # Resize for small overlay and convert BGRA to BGR
                display_frame = cv2.resize(img_array[:, :, :3], (320, 240))
                cv2.imshow(window_name, display_frame)
                cv2.waitKey(1)  # Non-blocking wait to update the window

                await asyncio.sleep(1 / 30)  # ~30 FPS

    async def capture_camera_frames(self):
        """Shared camera capture for both overlay and Gemini processing"""
        if self.video_mode != "camera":
            return

        cap = await asyncio.to_thread(cv2.VideoCapture, 0)

        while True:
            ret, frame = await asyncio.to_thread(cap.read)
            if not ret:
                break

            # Put frame in shared queue for overlay (non-blocking)
            if self.show_overlay:
                try:
                    self.camera_frame_queue.put_nowait(frame.copy())
                except asyncio.QueueFull:
                    # Skip frame if queue is full (overlay is behind)
                    pass

            await asyncio.sleep(1 / 30)  # Capture at 30 FPS

        cap.release()

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)
                self.speaking_lock = asyncio.Lock()  # Initialize the lock in async context

                # Initialize camera frame queue for shared camera access
                if self.video_mode == "camera" and self.show_overlay:
                    self.camera_frame_queue = asyncio.Queue(maxsize=10)

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())

                # Add camera capture task for shared camera access
                if self.video_mode == "camera":
                    if self.show_overlay:
                        tg.create_task(self.capture_camera_frames())
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())

                # Add overlay task for smooth video display
                tg.create_task(self.update_overlay())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())
                if self.auto_prompt:
                    tg.create_task(self.auto_prompt_loop())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            # Handle ExceptionGroup for Python 3.11+ and regular exceptions for older versions
            if hasattr(e, "exceptions") and sys.version_info >= (3, 11):
                # This is an ExceptionGroup
                self.audio_stream.close()
                traceback.print_exception(type(e), e, e.__traceback__)
            else:
                # Regular exception handling
                self.audio_stream.close()
                traceback.print_exception(type(e), e, e.__traceback__)
        finally:
            # Clean up OpenCV windows if overlay was shown
            if self.show_overlay:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="show camera/screen overlay window",
    )
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode, show_overlay=args.overlay)
    asyncio.run(main.run())
