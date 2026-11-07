import asyncio
import audioop
import os
import time
import serial
import numpy as np

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from openwakeword.model import Model
import openwakeword


load_dotenv()

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

MODEL = "gemini-3.8-live"

CONFIG = {
    "response_modalities": ["AUDIO"]
}

INPUT_RATE = 16000
OUTPUT_RATE = 24000

SILENCE_THRESHOLD = 500
SILENCE_DURATION = 2.0

CHUNK = 1024
WAKEWORD_CHUNK = 1280

ESP_PORT = "COM4"
ESP_BAUD = 921600

PACKET_SIZE = 512

START_BUFFER_SECONDS = 0.5

START_BUFFER_BYTES = int(
    OUTPUT_RATE
    * 2
    * START_BUFFER_SECONDS
)


openwakeword.utils.download_models()


def record_until_silence():

    import pyaudio

    p = pyaudio.PyAudio()

    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=INPUT_RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("Listening for your question...")

    frames = []

    silent_chunks = 0
    spoke_chunks = 0

    chunks_per_second = INPUT_RATE / CHUNK

    silence_limit = int(
        chunks_per_second
        * SILENCE_DURATION
    )

    while True:

        data = stream.read(
            CHUNK,
            exception_on_overflow=False
        )

        frames.append(data)

        volume = audioop.rms(
            data,
            2
        )

        if volume < SILENCE_THRESHOLD:

            silent_chunks += 1

        else:

            silent_chunks = 0
            spoke_chunks += 1

        if silent_chunks > silence_limit:

            print(
                "Silence detected, stopping."
            )

            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    min_speech_chunks = int(
        chunks_per_second * 0.5
    )

    if spoke_chunks < min_speech_chunks:

        return None

    return b"".join(frames)


def wait_for(
    esp,
    expected,
    timeout=5
):

    deadline = (
        time.monotonic()
        + timeout
    )

    while time.monotonic() < deadline:

        response = esp.read(1)

        if response == expected:

            return True

    return False


def send_audio_packet(
    esp,
    data
):

    packet = (
        b"\xAA\x55"
        + b"\x01"
        + len(data).to_bytes(
            2,
            "little"
        )
        + data
    )

    esp.write(packet)

    if not wait_for(
        esp,
        b"K"
    ):

        raise RuntimeError(
            "ESP32 did not acknowledge audio packet."
        )


def send_audio_data(
    esp,
    data
):

    for i in range(
        0,
        len(data),
        PACKET_SIZE
    ):

        chunk = data[
            i:i + PACKET_SIZE
        ]

        send_audio_packet(
            esp,
            chunk
        )


def finish_playback(esp):

    esp.write(
        b"\xAA\x55"
        + b"\x02"
        + b"\x00\x00"
    )

    if not wait_for(
        esp,
        b"R"
    ):

        raise RuntimeError(
            "ESP32 did not acknowledge END packet."
        )

    print(
        "Waiting for playback to finish..."
    )

    if not wait_for(
        esp,
        b"D",
        timeout=30
    ):

        raise RuntimeError(
            "ESP32 did not report playback complete."
        )


def normalize_audio(data):

    if len(data) % 2:

        data = data[:-1]

    if not data:

        return b""

    pcm = np.frombuffer(
        data,
        dtype="<i2"
    )

    pcm_float = pcm.astype(
        np.float32
    )

    peak = np.max(
        np.abs(pcm_float)
    )

    if peak > 0:

        # Leave a little headroom.
        gain = 28000.0 / peak

        pcm_float *= gain

    pcm = np.clip(
        pcm_float,
        -32768,
        32767
    ).astype(
        np.int16
    )

    return pcm.tobytes()


async def handle_conversation(
    esp
):

    mic_audio = record_until_silence()

    if mic_audio is None:

        print(
            "No speech detected."
        )

        return

    print(
        "Sending question to Gemini..."
    )

    # Queue containing PCM audio waiting
    # to be played.
    audio_queue = asyncio.Queue(
        maxsize=100
    )

    gemini_finished = False

    total_audio_bytes = 0

    first_audio_received = False

    playback_started = False

    async def gemini_receiver():

        nonlocal gemini_finished
        nonlocal total_audio_bytes
        nonlocal first_audio_received

        async with client.aio.live.connect(
            model=MODEL,
            config=CONFIG
        ) as session:

            await session.send_realtime_input(
                audio=types.Blob(
                    data=mic_audio,
                    mime_type=(
                        f"audio/pcm;rate={INPUT_RATE}"
                    )
                )
            )

            await session.send_realtime_input(
                audio_stream_end=True
            )

            try:

                async for msg in session.receive():

                    if not msg.data:

                        continue

                    audio_data = normalize_audio(
                        msg.data
                    )

                    if not audio_data:

                        continue

                    total_audio_bytes += len(
                        audio_data
                    )

                    first_audio_received = True

                    await audio_queue.put(
                        audio_data
                    )

            except errors.APIError as e:

                if "1000" not in str(e):

                    raise

        gemini_finished = True

        await audio_queue.put(
            None
        )

    async def playback_worker():

        nonlocal playback_started

        buffered_bytes = 0

        playback_buffer = []


        while buffered_bytes < START_BUFFER_BYTES:

            chunk = await audio_queue.get()

            if chunk is None:

                break

            playback_buffer.append(
                chunk
            )

            buffered_bytes += len(
                chunk
            )

        if not playback_buffer:

            return

        print(
            "Starting playback..."
        )

        playback_started = True

        for chunk in playback_buffer:

            send_audio_data(
                esp,
                chunk
            )

        playback_buffer.clear()


        while True:

            chunk = await audio_queue.get()

            if chunk is None:

                break

            send_audio_data(
                esp,
                chunk
            )

    receiver_task = asyncio.create_task(
        gemini_receiver()
    )

    playback_task = asyncio.create_task(
        playback_worker()
    )

    await receiver_task
    await playback_task

    if total_audio_bytes == 0:

        print(
            "Gemini returned no audio."
        )

        return

    duration = (
        total_audio_bytes
        / (OUTPUT_RATE * 2)
    )

    print(
        f"Gemini audio: {duration:.2f} sec"
    )

    finish_playback(
        esp
    )

    print(
        "Playback finished."
    )


async def main():

    oww_model = Model(
        wakeword_models=[
            "hey_jarvis_v0.1"
        ],
        inference_framework="onnx"
    )

    import pyaudio

    p = pyaudio.PyAudio()

    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=INPUT_RATE,
        input=True,
        frames_per_buffer=WAKEWORD_CHUNK
    )

    esp = serial.Serial(
        ESP_PORT,
        ESP_BAUD,
        timeout=0.1
    )

    time.sleep(1.0)

    esp.reset_input_buffer()

    print(
        "Listening for wake word..."
    )

    try:

        while True:

            audio_chunk = np.frombuffer(
                stream.read(
                    WAKEWORD_CHUNK,
                    exception_on_overflow=False
                ),
                dtype=np.int16
            )

            prediction = oww_model.predict(
                audio_chunk
            )

            for wakeword, score in prediction.items():

                if score > 0.7:

                    print(
                        f"Wake word '{wakeword}' detected!"
                    )

                    await handle_conversation(
                        esp
                    )

                    oww_model.reset()

                    time.sleep(1)

                    print(
                        "Listening for wake word..."
                    )

                    break

    finally:

        stream.stop_stream()
        stream.close()

        p.terminate()

        esp.close()


if __name__ == "__main__":

    asyncio.run(main())