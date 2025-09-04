"""
Gemini Live API Client - Hands-free voice and video conversation
Based on Google's official Live API example
"""

import asyncio
import base64
import io
import os
import traceback

import cv2
import dotenv
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

        print("🚀 Initializing Gemini Live Client...")
        print("🎤 Hands-free audio mode - speak naturally!")
        print("👀 Visual context from webcam")
        print("Press Ctrl+C to quit")

    def _get_frame(self, cap):
        """Capture and process a frame from the camera - exactly like Google's implementation"""
        # Read the frame
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
        """Continuously capture frames from camera - exactly like Google's implementation"""
        # This takes about a second, and will block the whole program
        # causing the audio pipeline to overflow if you don't to_thread it.
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)  # 0 represents the default camera

        try:
            while True:
                frame = await asyncio.to_thread(self._get_frame, cap)
                if frame is None:
                    break

                await asyncio.sleep(1.0)

                await self.out_queue.put(frame)
        finally:
            # Release the VideoCapture object
            cap.release()

    async def send_realtime(self):
        """Send data from queue to Gemini Live API - exactly like Google's implementation"""
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        """Continuously capture audio from microphone - exactly like Google's implementation"""
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
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        """Background task to reads from the websocket and write pcm chunks to the output queue - exactly like Google's implementation"""
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(text, end="")

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        """Play audio responses from Gemini - exactly like Google's implementation"""
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
        print("👀 Starting camera display...")

        while True:
            try:
                # Simple display without interfering with the main audio loop
                await asyncio.sleep(0.1)  # Just keep the task alive
            except asyncio.CancelledError:
                break

    async def send_initial_message(self):
        """Send initial greeting message"""
        await asyncio.sleep(2)  # Wait for everything to initialize
        await self.session.send(input="Hi! I can see you through the camera. Let's have a natural conversation. Feel free to speak naturally - I'm listening!", end_of_turn=True)

    async def run(self):
        """Main execution loop following Google's exact pattern"""
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                print("✅ Connected to Gemini Live API!")
                print("\n🎯 Ready for conversation!")
                print("💬 Speak naturally - I'm listening and watching!")

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                # Create tasks exactly like Google's implementation
                initial_task = tg.create_task(self.send_initial_message())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())
                tg.create_task(self.show_camera_feed())

                await initial_task

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
