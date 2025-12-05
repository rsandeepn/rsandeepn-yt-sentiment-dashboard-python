import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

print("Environment loaded!")

print("OpenAI key exists:", bool(os.getenv("OPEN_API_KEY")))
print("YouTube key exists:", bool(os.getenv("YOUTUBE_API_KEY")))

try:
    client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))
    print("OpenAI client created successfully")
except Exception as e:
    print("OpenAI Errors:", e)
