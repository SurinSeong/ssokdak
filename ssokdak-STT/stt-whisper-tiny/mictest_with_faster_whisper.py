import os
import time
from faster_whisper import WhisperModel # faster_whisper
import numpy as np
import soundfile as sf
import noisereduce as nr


# 커스텀한 STT 모델
class Cumtom_whisper:

    def __init__(self):
        '''
        최대 4배 빠른 faster whisper를 사용하여 cpu로 저장된 wav파일에 STT 수행
        
        model_size : tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, large-v3, or large
        '''
        # 환경 설정(Window 아나콘다 환경에서 아래 코드 실행 안하면 에러남)
        try: os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
        except Exception as e: print(f'os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true" 실행해서 발생한 에러. 하지만 무시하고 진행: {e}')

        try: os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        except Exception as e: print(f'os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 실행해서 발생한 에러. 하지만 무시하고 진행: {e}')


    def set_model(self, model_name):
        '''
        model_size : tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, large-v3, or large
        '''
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
        print(f'STT 모델: {model_name}')


    def run(self, audio, language=None):
        '''
        저장된 tmp.wav를 불러와서 STT 추론 수행 (노이즈도 제거)

        audio : wav파일의 경로 or numpy로 변환된 오디오 파일 소스
        language : ko, en 등 언어 선택 가능. 선택하지 않으면 언어 분류 모델 내부적으로 수행함
        '''
        start = time.time()

        # 입력 처리
        if isinstance(audio, str):
            audio, sr = sf.read(audio)
            assert sr == 16000, f"샘플링 레이트는 16000Hz여야 함. 현재:{sr}"
            audio = audio.astype(np.float32)

        elif isinstance(audio, np.ndarray):
            sr = 16000
            audio = audio.astype(np.float32)
        else:
            raise ValueError("audio must be either a numpy array or a valid file path string.")
        
        # 노이즈 제거
        noise_clip = audio[:int(sr * 0.5)]    # 앞 0.5초를 노이즈로 간주
        denoised = nr.reduce_noise(y=audio, y_noise=noise_clip, sr=sr, prop_decrease=0.2)
        self.last_denoised_audio = denoised  # 최근 오디오 저장

        # 추론
        segments, info = self.model.transcribe(audio, beam_size=5, word_timestamps=True, language=language, vad_filter=True, no_speech_threshold=0.6)
        
        # 결과 후처리
        dic_list = []

        for segment in segments:
            if segment.no_speech_prob > 0.6:
                continue # 말을 안했을 확률이 크다고 감지되면 무시

            for word in segment.words:
                _word = word.word
                _start = round(word.start, 2)
                _end = round(word.end, 2)
                dic_list.append([_word, _start, _end])

        # 시간 계산
        self.spent_time = round(time.time()-start, 2)
        
        # 텍스트 추출
        result_txt = self._make_txt(dic_list)
        return dic_list, result_txt

    def _make_txt(self, dic_list):
        '''
        [word, start, end]에서 word만 추출하여 txt로 반환
        '''
        result_txt = ''
        for dic in dic_list:
            txt = dic[0]
            result_txt = f'{result_txt}{txt}'
        return result_txt
    
    def save_last_audio(self, filename="denoised_output.wav"):
        if self.last_denoised_audio is None:
            print("⚠️ 아직 STT가 수행되지 않았습니다.")
            return
        
        sf.write(filename, self.last_denoised_audio, 16000)
        print(f"✅ 노이즈 제거된 오디오가 저장되었습니다: {filename}")
    

# 웨이크워드 탐지 함수
def contains_wakeword(text, wakewords):
    for word in wakewords:
        if word in text:
            return True
        
    return False


custom_wakewords = ["도서관"]
wakeword_detected = False

custom_model = Cumtom_whisper()
custom_model.set_model('small')

audio_data = "./data/B-C9884F-C-NULNU-s0mN-C711535-S.wav"

print('wav파일 입력')

# 1. STT 수행
dic_list, result_txt = custom_model.run(audio_data)

# 노이즈 제거된 오디오 저장
custom_model.save_last_audio()

# 2. 웨이크워드 탐지
wakeword_detected = contains_wakeword(result_txt, custom_wakewords)

# 3. 대화 상태 업데이트
if wakeword_detected:
    session_state["is_talking"] = True
    # → LLM에게 전체 result_txt 전달
    response = send_to_llm(result_txt)

elif session_state.get("is_talking", False):
    # 웨이크워드는 없지만 대화는 진행 중
    response = send_to_llm(result_txt)

else:
    # 대화 중이 아니면 LLM 호출 안 함
    response = None