import requests
import os
from dotenv import load_dotenv

load_dotenv()

X_SUP_API_KEY = os.getenv("X_SUP_API_KEY")
BASE_URL = os.getenv("BASE_URL")
SAVE_FOLDER_PATH = os.getenv("SAVE_FOLDER_PATH")

def make_voice_with_supertone(voice_id, idx, text, style):
    url = BASE_URL + f"/v1/text-to-speech/{voice_id}"
    headers = {
        "x-sup-api-key": X_SUP_API_KEY
    }

    payload = {
        "text": text,
        "language": "ko",
        "style": style,
        "model": "sona_speech_1",
        "output_format": "wav",
        "voice_settings": {
            "pitch_shift": 0,
            "pitch_variance": 1,
            "speed": 1
        }
    }

    response = requests.post(url, json=payload, headers=headers)


    with open(f"{SAVE_FOLDER_PATH}/result_{idx}.wav", "wb") as f:   # 원하는 파일명 지정
        f.write(response.content)



text_list = [
    "sad|속상했겠다. 어떤 일 때문에 싸운 거야?",
    "sad|그건 정말 기분 나빴겠다.",
    "curious|음, 어떻게 화냈어?",
    "neutral|아 그렇구나. 친구가 깜짝 놀랐을 수도 있겠다.",
    "neutral|화가 나서 그랬을 수도 있지만 다음에는 생각을 또박또박 말해주면 친구도 더 잘 이해할 수 있을 거야.",
    "neutral|그럴 수 있어. 그 마음을 나도 이해해",
    "happy|잘했다고 말해주고 싶어. 사과할 줄 아는 사람이 진짜 멋진 사람이거든!",
    "happy|또 속상한 일이 있으면 나한테 말해줘!"
]

VOICE_ID = "2d5a380030e78fcab0c82a"

for idx, text in enumerate(text_list, 1):
    style = text.split("|")[0]
    main_text = text.split("|")[1]
    make_voice_with_supertone(VOICE_ID, idx, main_text, style)