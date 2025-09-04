"""
Gemini Live API Client - Hands-free voice and video conversation
Fixed to prevent automatic responses by waiting for user input first
"""

import asyncio
import base64
import io
import os
import time
import traceback

import cv2
import dotenv
import numpy as np
import PIL.Image
import pyaudio
from google import genai
from google.genai import types

# Load environment variables
dotenv.load_dotenv()

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.5-flash-preview-native-audio-dialog"

DEFAULT_MODE = "camera"

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=os.environ.get("GOOGLE_API_KEY"),
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
    system_instruction=("You are a helpful AI assistant having a natural conversation. Respond naturally to what the user says and comment on what you see in the camera feed. Keep your responses conversational and engaging. Be concise but friendly."),
)

pya = pyaudio.PyAudio()


class GeminiLiveClient:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode

        self.audio_in_queue = None
        self.out_queue = None

        self.session = None
        self.audio_stream = None

        # State tracking for conversation turns
        self.is_gemini_speaking = False
        self.user_speaking = False
        self.last_user_activity = 0
        self.silence_threshold = 1.5  # seconds of silence before considering user done
        self.camera_cap = None
        self.volume_threshold = 80  # Lowered threshold - adjust this based on your microphone sensitivity
        self.turn_complete = False  # Track if user turn is complete
        self.waiting_for_user = True  # Track if we're waiting for user input
        self.receive_task = None  # Track the receive task
        self.first_user_input = False  # Track if user has spoken for the first time

        print("🚀 Initializing Gemini Live Client...")
        print("🎤 Hands-free audio mode - speak naturally!")
        print("👀 Visual context from webcam")
        print("💡 Start speaking to begin the conversation!")
        print("Press Ctrl+C to quit")

    def _get_frame(self, cap):
        """Capture and process a frame from the camera"""
        ret, frame = cap.read()
        if not ret:
            return None
        # Convert BGR to RGB color space
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def initialize_camera(self):
        """Initialize camera and show test frame"""
        print("📷 Initializing camera...")
        self.camera_cap = await asyncio.to_thread(cv2.VideoCapture, 0)

        # Test camera
        ret, frame = await asyncio.to_thread(self.camera_cap.read)
        if ret:
            print("✅ Camera initialized successfully!")
            # Show a small preview
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(frame_rgb)
            img.thumbnail([320, 240])
            print(f"📐 Camera resolution: {img.size}")
        else:
            print("❌ Failed to initialize camera")
            self.camera_cap = None

    async def capture_single_frame(self):
        """Capture a single frame when user finishes speaking"""
        if self.camera_cap is None:
            return None

        frame = await asyncio.to_thread(self._get_frame, self.camera_cap)
        return frame

    async def send_realtime(self):
        """Send data from queue to Gemini Live API using new methods"""
        while True:
            msg = await self.out_queue.get()
            print(f"🔍 DEBUG: Sending message type: {msg.get('mime_type')}")

            # Use the new send_realtime_input method
            if msg.get("mime_type") == "audio/pcm":
                print("🔍 DEBUG: Sending audio chunk")
                await self.session.send_realtime_input(audio=types.Blob(data=msg["data"], mime_type="audio/pcm;rate=16000"))
            elif msg.get("mime_type") == "image/jpeg":
                print("🔍 DEBUG: Sending image")
                await self.session.send_realtime_input(image=types.Blob(data=base64.b64decode(msg["data"]), mime_type="image/jpeg"))

    def calculate_volume(self, audio_data):
        """Calculate RMS volume from audio data"""
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        # Calculate RMS (Root Mean Square) volume with safety check
        mean_square = np.mean(audio_array**2)
        if mean_square < 0:
            mean_square = 0
        rms = np.sqrt(mean_square)
        return rms

    async def start_receive_task(self, task_group):
        """Start the receive task when user first speaks"""
        if self.receive_task is None:
            print("🔍 DEBUG: Starting receive_audio task for first time...")
            self.receive_task = task_group.create_task(self.receive_audio())

    async def listen_audio(self, task_group):
        """Continuously capture audio from microphone with improved voice activity detection"""
        mic_info = pya.get_default_input_device_info()
        print(f"🎤 Using microphone: {mic_info['name']}")

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

        silence_start = None
        print("🎧 Audio input ready - speak to start conversation...")
        print(f"🔊 Volume threshold: {self.volume_threshold} (speak loudly to trigger)")

        # Add volume monitoring
        volume_samples = []
        last_volume_print = 0

        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)

            # Calculate volume using RMS
            volume = self.calculate_volume(data)
            volume_samples.append(volume)

            # Print volume levels every 2 seconds for debugging
            current_time = time.time()
            if current_time - last_volume_print > 2:
                avg_volume = sum(volume_samples) / len(volume_samples)
                max_volume = max(volume_samples)
                print(f"🔊 Audio levels - Avg: {avg_volume:.1f}, Max: {max_volume:.1f}, Threshold: {self.volume_threshold}")
                volume_samples = []
                last_volume_print = current_time

            # Skip audio processing if Gemini is speaking or not waiting for user
            if self.is_gemini_speaking or not self.waiting_for_user:
                continue

            is_speaking = volume > self.volume_threshold

            if is_speaking:
                # User is speaking
                if not self.user_speaking:
                    self.user_speaking = True
                    self.turn_complete = False
                    self.waiting_for_user = False  # Stop waiting, user is speaking
                    print(f"🎤 User speaking... (volume: {volume:.1f})")

                    # If this is the first user input, start the receive task
                    if not self.first_user_input:
                        self.first_user_input = True
                        print("🚀 First user input detected! Starting conversation...")
                        await self.start_receive_task(task_group)

                self.last_user_activity = current_time
                silence_start = None  # Reset silence timer

                # Send audio in real-time
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

            elif self.user_speaking and not self.turn_complete:
                # User was speaking but now silent
                if silence_start is None:
                    silence_start = current_time

                # Check if silence has lasted long enough
                if current_time - silence_start > self.silence_threshold:
                    print("📸 User finished speaking - capturing image...")

                    # Capture and send image
                    frame = await self.capture_single_frame()
                    if frame:
                        await self.out_queue.put(frame)

                    # Mark turn as complete
                    self.turn_complete = True
                    self.user_speaking = False
                    silence_start = None
                    print("✅ Turn complete - waiting for Gemini response...")
                else:
                    # Still in silence period, keep sending audio
                    await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
            else:
                # Not speaking or turn already complete - don't send audio
                pass

    async def receive_audio(self):
        """Background task to read from the websocket and write pcm chunks to the output queue"""
        print("🔍 DEBUG: Starting receive_audio task")

        while True:
            print("🔍 DEBUG: Waiting for turn from session.receive()...")
            turn = self.session.receive()
            print("🔍 DEBUG: Got turn from session.receive()")

            self.is_gemini_speaking = True  # Mark that Gemini is about to speak
            self.waiting_for_user = False  # Not waiting for user while Gemini speaks
            print("🤖 Gemini responding...")

            async for response in turn:
                print(f"🔍 DEBUG: Processing response: {type(response)}")
                if data := response.data:
                    print("🔍 DEBUG: Got audio data")
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(f"📝 DEBUG: Got text: {text}")
                    print(text, end="")

            # Turn complete - Gemini finished speaking
            self.is_gemini_speaking = False
            self.turn_complete = False  # Reset for next user turn
            self.waiting_for_user = True  # Now waiting for user input
            print("\n✅ Ready to listen...")

            # Clear any remaining audio in the queue for interruptions
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        """Play audio responses from Gemini"""
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

    async def show_camera_feed(self):
        """Display camera feed with status information"""
        print("📹 Camera feed active...")
        while True:
            try:
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break

    async def run(self):
        """Main execution loop with improved conversation flow"""
        try:
            # Initialize camera first
            await self.initialize_camera()

            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                print("✅ Connected to Gemini Live API!")
                print("\n🎯 Ready for conversation!")
                print("💬 Start speaking - I'm listening and watching!")
                print("⚠️  NO initial message sent - waiting for your input!")

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                # Create basic tasks (NO receive_audio task yet)
                print("🔍 DEBUG: Creating basic tasks...")
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio(tg))  # Pass task group for receive task creation
                tg.create_task(self.play_audio())
                tg.create_task(self.show_camera_feed())

                print("🔍 DEBUG: All tasks created, waiting for user input...")

                # Keep running until interrupted
                while True:
                    await asyncio.sleep(1)

        except asyncio.CancelledError:
            print("\n👋 Shutting down gracefully...")
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
            traceback.print_exc()
        finally:
            # Cleanup
            if self.audio_stream:
                self.audio_stream.close()
            if self.camera_cap:
                self.camera_cap.release()
            print("✅ Cleanup complete")


async def main():
    """Main entry point"""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in environment variables")
        print("Please create a .env file with your Google API key:")
        print("GOOGLE_API_KEY=your_api_key_here")
        print("\nGet your API key from: https://aistudio.google.com/app/apikey")
        return

    client_instance = GeminiLiveClient(video_mode="camera")
    await client_instance.run()


if __name__ == "__main__":
    print("🎬 Starting Gemini Live Client...")
    asyncio.run(main())
