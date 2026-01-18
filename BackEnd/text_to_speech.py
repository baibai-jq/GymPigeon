import os
import threading
import uuid
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()

# Initialize client safely
try:
    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
except Exception as e:
    print(f"ElevenLabs Warning: {e}")
    client = None

def speak(text):
    """
    Generates and plays audio in a background thread using macOS 'afplay'.
    This prevents the server from freezing during video processing.
    """
    if not client or not text:
        return

    def _audio_task():
        try:
            # 1. Generate Audio Stream
            audio_generator = client.text_to_speech.convert(
                text=text,
                voice_id="JBFqnCBsd6RMkjVDRZzb",
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
            )

            # 2. Save to a unique temporary file
            # We use uuid so multiple messages don't overwrite each other
            audio_bytes = b"".join(audio_generator)
            filename = f"temp_speech_{uuid.uuid4().hex}.mp3"
            
            with open(filename, "wb") as f:
                f.write(audio_bytes)

            # 3. Play using macOS native player
            # (Works reliably without installing 'mpv')
            os.system(f"afplay {filename}")

            # 4. Cleanup
            if os.path.exists(filename):
                os.remove(filename)

        except Exception as e:
            print(f"TTS Error: {e}")

    # Run in a separate thread to not block the main program
    thread = threading.Thread(target=_audio_task)
    thread.daemon = True
    thread.start()

# --- THIS IS THE PART YOU ASKED FOR ---
if __name__ == "__main__":
    print("Testing Text-to-Speech...")
    speak("System check complete. Audio is working.")
    
    # Keep the script alive long enough for the background thread to finish
    import time
    time.sleep(5)
    print("Test finished.")