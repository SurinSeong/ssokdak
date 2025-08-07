from vertexai.preview.generative_models import GenerativeModel
import vertexai
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = \
"ADD-YOUR-KEY.json"

vertexai.init(project="ssokdak", location="asia-northeast3")

user_message = "인공지능에 대해 한 문장으로 말하세요."
model = GenerativeModel(model_name='gemini-2.5-flash')    # gemini-2.5-flash
resp = model.generate_content(user_message)
print(resp.text)

# from google import genai

# client = genai.Client(
#   vertexai=True, project="ssokdak", location="global",
# )
# # If your image is stored in Google Cloud Storage, you can use the from_uri class method to create a Part object.
# model = "gemini-2.5-flash"
# response = client.models.generate_content(
#   model=model,
#   contents=[
#     "인공지능에 대해 한 문장으로 말하세요.",
#   ],
# )
# print(response.text, end="")