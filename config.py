from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHUNK_SIZE = os.getenv("CHUNK_SIZE")
BASE_URL = os.getenv("BASE_URL")
