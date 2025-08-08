# 모델 로드하기 위해서는 어떤 라이브러리 (torch 일듯)가 필요하다 -> 테스트 실패..

import whisper
from . import save_audio

# Whisper 모델 로드
model = whisper.load_model("base")

# AudioSave 클래스 인스턴스 생성 및 녹음 시작
audio_save = save_audio.AudioSave()
audio_save.run()

# 일정 시간 녹음 후 중지
import time

time.sleep(5)  # 5초 동안 녹음
audio_data, sr = audio_save.stop()  # numpy 배열과 샘플링 레이트 반환

# Whisper 모델을 사용하여 녹음된 오디오 데이터를 처리
result = model.transcribe(audio_data)
print('오디오 소스 입력')
print(result)
print('')
print('wav파일 입력')
result = model.transcribe('output_whisper.wav')
print(result)