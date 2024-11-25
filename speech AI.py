import deepspeech
import pyaudio
import wave
import numpy as np
from queue import Queue
import threading


MODEL_PATH = 'path_to_your_model/deepspeech-0.9.3-models.pbmm'
SCORER_PATH = 'path_to_your_model/deepspeech-0.9.3-models.scorer'

model = deepspeech.Model(MODEL_PATH)
model.enableExternalScorer(SCORER_PATH)

RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
FRAMES_PER_BUFFER = 1024

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=FRAMES_PER_BUFFER)


q = Queue()

def audio_callback(in_data, frame_count, time_info, status):
    q.put(np.frombuffer(in_data, dtype=np.int16))
    return (None, pyaudio.paContinue)


stream.start_stream()


def process_audio():
    audio_data = np.array([])

    while True:
        if not q.empty():
            audio_data = np.concatenate((audio_data, q.get()))

            
            if len(audio_data) > RATE // 10:  
                text = model.stt(audio_data)
                print(f"識別結果: {text}")
                audio_data = np.array([])


thread = threading.Thread(target=process_audio)
thread.daemon = True
thread.start()


stream.set_stream_callback(audio_callback)


try:
    while True:
        pass
except KeyboardInterrupt:
    stream.stop_stream()
    stream.close()
    p.terminate()
