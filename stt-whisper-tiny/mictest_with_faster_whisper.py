import os
import time
from faster_whisper import WhisperModel # faster_whisper
import numpy as np
import soundfile as sf
import noisereduce as nr


# 노이즈 제거
def reduce_noise(buffer, sample_rate):
    # 버퍼를 하나의 오디오 데이터로 결합한다.
    audio_data = np.frombuffer(b''.join(buffer), dtype=np.int16)

    # 노이즈 감소
    reduced_noise = nr.reduce_noise(y=audio_data,
                                    sr=sample_rate,
                                    prop_decrease=0.0)

    return reduced_noise


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
        print(f'STT 모델 변경: {model_name}')

    def set_wakewords(self, wakewords):
        self.wakewords = wakewords
        print(f"✅ 웨이크워드 설정됨: {wakewords}")

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
        wakeword_detected = False    # 감지 유무

        for segment in segments:
            if segment.no_speech_prob > 0.6:
                continue # 말을 안했을 확률이 크다고 감지되면 무시

            segment_text = segment.text.strip()

            # 웨이크워드 탐지
            for w in self.wakewords:
                if w in segment_text:
                    print(f"✅ [웨이크워드 '{w}' 감지됨!] → 즉시 반응 가능")
                    wakeword_detected = True
                    break
            
            # if wakeword_detected:
            #     # llm 모델을 준비시킨다.
            #     pass

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

custom_model = Cumtom_whisper()
custom_model.set_model('base')
custom_model.set_wakewords(custom_wakewords)

audio_data = "./data/_01_F_HYH00_10___00802.wav"

print('wav파일 입력')
dic_list, result_txt = custom_model.run(audio_data)

# 노이즈 제거된 오디오 저장
custom_model.save_last_audio()

print(result_txt)