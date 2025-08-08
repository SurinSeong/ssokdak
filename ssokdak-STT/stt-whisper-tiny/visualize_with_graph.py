import numpy as np
import matplotlib.pyplot as plt
import wave
import sys

def wav_to_wave(path):
    wav_file = wave.open(path, 'r')
    signal = wav_file.readframes(-1)
    signal = np.frombuffer(signal, dtype=np.int16)
    sample_rate = wav_file.getframerate()
    time = np.linspace(0, len(signal) / sample_rate, num=len(signal))
    return signal, sample_rate, time

def visualize_audio(audio, sample_rate, label):
    '''
    음성파일을 그래프로 나타냄
    
    audio : .wav파일의 경로 or 넘파이 오디오 배열
    sample_rate : 넘파이 오디오 배열 입력 시 필요함, .wav 경로 입력 시 None으로 입력할 것
    label : 그래프에 표시할 레이블
    '''
    # .wav 파일의 경로를 입력했을 경우
    if sample_rate == None:
        signal, sample_rate, time = wav_to_wave(audio)
    # 넘파이 배열을 입력했을 경우
    else:
        signal = audio
        time = np.linspace(0, len(signal) / sample_rate, num=len(signal))
    
    print(signal)
    # 그래프 그리기
    plt.figure(figsize=(15, 5))
    plt.plot(time, signal)
    plt.title(label)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()    

# 예시
file_path_list = ["C:/Users/SSAFY/Desktop/git-repos/ssokdak/stt-whisper-tiny/output.wav", "C:/Users/SSAFY/Desktop/git-repos/ssokdak/stt-whisper-tiny/output_remove_noise.wav"]
label_list = ["with noise", "without_noise"]

for path, label in zip(file_path_list, label_list):
    visualize_audio(audio=path, sample_rate=None, label=label)