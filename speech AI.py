import deepspeech
import pyaudio
import wave
import numpy as np
from queue import Queue
import threading

# 配置 DeepSpeech 模型
MODEL_PATH = 'path_to_your_model/deepspeech-0.9.3-models.pbmm'
SCORER_PATH = 'path_to_your_model/deepspeech-0.9.3-models.scorer'

# 初始化 DeepSpeech 模型
model = deepspeech.Model(MODEL_PATH)
model.enableExternalScorer(SCORER_PATH)

# 配置麥克風設置
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
FRAMES_PER_BUFFER = 1024

# 音頻流設置
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=FRAMES_PER_BUFFER)

# 創建用於緩衝音頻數據的隊列
q = Queue()

# 實時語音識別回調
def audio_callback(in_data, frame_count, time_info, status):
    q.put(np.frombuffer(in_data, dtype=np.int16))
    return (None, pyaudio.paContinue)

# 開始音頻流
stream.start_stream()

# 開啟語音識別線程
def process_audio():
    audio_data = np.array([])

    while True:
        if not q.empty():
            audio_data = np.concatenate((audio_data, q.get()))

            # 若收集到足夠的音頻數據
            if len(audio_data) > RATE // 10:  # 這裡設定每10毫秒一個批次
                text = model.stt(audio_data)
                print(f"識別結果: {text}")
                audio_data = np.array([])

# 啟動語音識別處理線程
thread = threading.Thread(target=process_audio)
thread.daemon = True
thread.start()

# 設置音頻流回調
stream.set_stream_callback(audio_callback)

# 讓程式繼續運行，直到用戶手動停止
try:
    while True:
        pass
except KeyboardInterrupt:
    stream.stop_stream()
    stream.close()
    p.terminate()
