import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

# 1. Load Env
load_dotenv()
api_key = os.getenv("ELEVENLABS_API_KEY")

if not api_key:
    print("ERROR: API Key not found in .env")
    exit()

print("1. Authenticating...")
client = ElevenLabs(api_key=api_key)

try:
    print("2. Generating audio...")
    # Generate the audio generator object
    audio_generator = client.text_to_speech.convert(
        text="This is a test of the emergency broadcast system.",
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    
    # Consume the generator to get the actual bytes
    # (If the API fails, it usually fails here)
    audio_bytes = b"".join(audio_generator)
    print(f"   Success! Received {len(audio_bytes)} bytes.")

    # 3. Save to file (This proves the data is real)
    filename = "temo_output.mp3"
    with open(filename, "wb") as f:
        f.write(audio_bytes)
    print(f"3. Saved to {filename}")

    # 4. Play using macOS native player (No 'mpv' required)
    print("4. Playing audio via 'afplay'...")
    os.system(f"afplay {filename}") 
    print("Done.")

except Exception as e:
    print(f"\nCRASH: {e}")