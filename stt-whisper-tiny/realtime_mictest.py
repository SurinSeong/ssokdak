# 노이즈 제거 X

import speech_recognition as sr
import wave
import io
import noisereduce as nr
import numpy as np


# 오디오 데이터를 wav 파일로 저장하는 함수
def save_buffer_to_wav(buffer, sample_rate, sample_width, filename):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # 모노
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(buffer))


# 마이크에서 음성을 스트리밍하는 함수
def record_audio(filename="output_remove_noise.wav"):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone(sample_rate=16000)
    buffer = []
    chunk_size = 16000
    num_chunks = 5

    with microphone as source:
        print("Adjusting for ambient noise..."); recognizer.adjust_for_ambient_noise(source)  # 주변 소음을 기준으로 에너지 임계값 조정
        
        print("Recording...")
        recognizer.energy_threshold = recognizer.energy_threshold + 100

        while len(buffer) < num_chunks:
            buffer.append(source.stream.read(chunk_size))

    # 버퍼를 하나의 오디오 데이터로 결합한다.
    audio_data = np.frombuffer(b''.join(buffer), dtype=np.int16)

    # 노이즈 감소
    reduced_noise = nr.reduce_noise(y=audio_data,
                                    sr=microphone.SAMPLE_RATE,
                                    prop_decrease=0.0)
    
    # 노이즈 감소된 데이터를 다시 버퍼로 변환
    buffer = [reduced_noise.tobytes()]

    sample_rate = microphone.SAMPLE_RATE
    sample_width = microphone.SAMPLE_WIDTH
    save_buffer_to_wav(buffer, sample_rate, sample_width, filename)
    print("Audio saved to {}".format(filename))

    return np.frombuffer(buffer[0], dtype=np.int16), sample_rate


if __name__ == "__main__":
    record_audio()