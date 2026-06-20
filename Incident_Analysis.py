# this will generate Report based on the completed_incidents.json file
import json
from groq import Groq 
import os
from dotenv import load_dotenv
load_dotenv()

def GROQ_report(incident):
    API_KEY = os.environ.get("GROQ_API_AI_SAFETY_REPORT")
    if not API_KEY:
        raise Exception("Please set the GROQ_API_KEY environment variable")
    client = Groq(
        api_key=API_KEY
    )

    # with open("completed_incidents.json","r") as f:
    #     incidents = json.load(f)

    prompt = f"""
    You are a construction safety officer.

    Analyze this incident:

    {incident}

    Give:

    1. Risk explanation
    2. Potential consequences
    3. Recommended action

    Keep answer under 150 words.
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role":"user","content":prompt}
        ]
    )

    analysis = response.choices[0].message.content
    return analysis