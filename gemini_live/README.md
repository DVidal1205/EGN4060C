# Gemini Live API Client

A **hands-free, real-time conversational AI client** that uses Google's official Gemini Live API with WebSockets for seamless voice and video interactions.

## 🌟 Features

-   🎤 **Hands-free audio streaming** - No buttons, just speak naturally!
-   📹 **Live webcam context** - Gemini can see and comment on your environment
-   🔄 **Real-time bidirectional audio** - Native audio processing with Gemini
-   🧠 **Smart audio management** - Microphone automatically mutes when Gemini speaks
-   💬 **Natural conversations** - Powered by `gemini-2.5-flash-preview-native-audio-dialog`
-   🎛️ **Visual feedback** - Real-time status indicators and camera feed

## 🚀 What's New (Live API Implementation)

This client now uses the **official Gemini Live API** with WebSockets for:

-   ✅ **True real-time streaming** (not request/response)
-   ✅ **Native audio processing** (24kHz output from Gemini)
-   ✅ **Duplex communication** (simultaneous send/receive)
-   ✅ **Voice Activity Detection** built into the model
-   ✅ **Lower latency** conversations

## 📋 Setup

### 1. Install System Dependencies (Linux/Debian)

```bash
sudo apt-get update
sudo apt-get install -y python3-dev portaudio19-dev
```

### 2. Install Python Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Get Google API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Copy the API key

### 4. Configure Environment

Create a `.env` file in the project root:

```bash
# .env
GOOGLE_API_KEY=your_google_api_key_here
```

## 🎯 Usage

### Run the Client

```bash
cd gemini_live
python run.py
```

### How to Use

1. **Start the application** - The webcam feed opens with status indicators
2. **Speak naturally** - No buttons to press, just start talking!
3. **See the magic** - Gemini responds with voice and can comment on what it sees
4. **Natural conversation** - The microphone intelligently mutes when Gemini speaks
5. **Press 'Q'** - To quit the application

## 🔧 Technical Implementation

### Audio Pipeline

-   **Input**: 16kHz, 16-bit, mono PCM from microphone
-   **Output**: 24kHz, 16-bit, mono PCM from Gemini
-   **Streaming**: 100ms audio chunks via WebSocket
-   **Processing**: Threaded audio capture and playback

### Smart Audio Management

-   ✅ **Auto-muting**: Microphone stops capturing when Gemini speaks
-   ✅ **Real-time buffering**: Continuous audio streaming without gaps
-   ✅ **Echo prevention**: No feedback loops or audio conflicts

### Visual Context

-   **Camera feed**: Live webcam at ~30 FPS
-   **Image compression**: JPEG @ 75% quality for efficiency
-   **Context updates**: Fresh images sent after each Gemini response
-   **Status overlay**: Real-time microphone and speaking indicators

### WebSocket Integration

Based on the [official Live API documentation](https://ai.google.dev/gemini-api/docs/live):

-   Uses `client.aio.live.connect()` for WebSocket sessions
-   Streams with `send_realtime_input()` and `receive()`
-   Supports multimodal input (audio + image + text)
-   Native audio models for best quality

## 🎛️ Controls & Indicators

### Visual Status Indicators

-   **Green "Listening..."** - Microphone is active, speak freely
-   **Red "Gemini Speaking"** - AI is responding, microphone muted
-   **Green MIC circle** - Microphone available
-   **Gray MIC circle** - Microphone muted (Gemini speaking)

### Keyboard Controls

-   **Q** - Quit the application (only control needed!)

## 🛠️ System Requirements

### Hardware

-   **Microphone** - Any USB or built-in microphone
-   **Speakers/Headphones** - For audio output
-   **Webcam** - USB or built-in camera
-   **Internet** - Stable connection for Live API

### Software

-   **Python 3.11+**
-   **Linux/macOS/Windows** (tested on Linux)
-   **Audio drivers** - ALSA/PulseAudio (Linux) or system default

## 🎨 Model Configuration

The client uses the **native audio model** for optimal performance:

```python
MODEL = "gemini-2.5-flash-preview-native-audio-dialog"
```

**Why native audio?**

-   🗣️ **Natural speech patterns** - Better prosody and emotion
-   🌍 **Multilingual support** - Superior non-English performance
-   🧠 **Advanced features** - Emotion-aware responses and "thinking"
-   ⚡ **Real-time optimized** - Built for Live API streaming

## 🔍 Troubleshooting

### Audio Issues

```bash
# Check audio devices
arecord -l  # List recording devices
aplay -l    # List playback devices

# Test microphone
arecord -d 3 test.wav && aplay test.wav
```

### Camera Issues

```bash
# List video devices
ls /dev/video*

# Test camera
ffmpeg -f v4l2 -i /dev/video0 -frames 1 test.jpg
```

### API Issues

-   ✅ Verify API key is correct and active
-   ✅ Check internet connection stability
-   ✅ Ensure sufficient API quota
-   ✅ Try different model if issues persist

### Performance Optimization

-   **Lower camera resolution** - Edit `capture_frame()` for smaller images
-   **Adjust audio buffer** - Modify `CHUNK` size for different latency
-   **Network optimization** - Use wired connection for best performance

## 🚀 Advanced Features

### Conversation Context

-   **Visual memory** - Gemini remembers what it sees between interactions
-   **Natural flow** - Maintains conversation context across turns
-   **Environment awareness** - Comments on changes in your surroundings

### Real-time Processing

-   **Continuous streaming** - No start/stop recording needed
-   **Low latency** - ~100-200ms response time
-   **Parallel processing** - Audio and video handled simultaneously

## 🔮 Future Enhancements

-   [ ] **Voice Activity Detection** tuning
-   [ ] **Multi-language support** configuration
-   [ ] **Conversation recording** and playback
-   [ ] **Custom system prompts** via config file
-   [ ] **Screen sharing** capability
-   [ ] **Multiple camera** support
-   [ ] **Audio effects** and noise reduction

## 📚 Documentation References

-   [Gemini Live API Docs](https://ai.google.dev/gemini-api/docs/live)
-   [Native Audio Models](https://ai.google.dev/gemini-api/docs/live#choose-audio-generation)
-   [WebSocket API Reference](https://ai.google.dev/gemini-api/docs/live#websockets-api)

## 📄 License

This project is open source and available under the MIT License.
