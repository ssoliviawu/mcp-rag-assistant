import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url =os.getenv("OPENAI_BASE_URL")
)

def generate(
    system_prompt: str,
    user_prompt: str,
    model = os.environ["OPENAI_MODEL"],
) -> str:

    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=user_prompt,
    )

    return response.output_text