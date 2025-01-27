

speech = """TO EXPLORE NEW AREAS, YOU CAN ROTATE LEFT BY 90 DEGREES AND MOVE FORWARD. HERE'S HOW IT WORKS STEP-BY-STEP:

1. **CURRENT POSITION**: YOU'RE AT COORDINATES (X,Y) = (-8,6), FACING NORTH. THE SURROUNDING AREA HAS OBSTACLES (#) ON BOTH SIDES.

2. **ROTATION**: TURN LEFT BY 90 DEGREES TO FACE WEST. THIS CHANGES YOUR DIRECTION FROM NORTH TO WEST.

3. **MOVEMENT**: MOVE FORWARD ONE STEP IN THE WEST DIRECTION, WHICH UPDATES YOUR POSITION TO (X,Y) = (-9,6).

THIS MOVEMENT TAKES YOU INTO A LESS EXPLORED AREA WITH FEWER OBSTACLES, AIDING IN EFFECTIVE EXPLORATION OF THE MAP."""

from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key = api_key)
speech_file_path = Path(__file__).parent / "speech.mp3"
response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input=speech,
)
response.stream_to_file(speech_file_path)

