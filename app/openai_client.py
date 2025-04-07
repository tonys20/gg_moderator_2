import os
from openai import OpenAI
import logging
from dotenv import load_dotenv
from misc import Colors

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

def get_openai_response(prompt):
    client = OpenAI(
        api_key=os.getenv("OPENAI_KEY"),
        timeout=60,
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    try:
        stream = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            stream=False
        )
        return stream
    except Exception as e:
        if hasattr(e, "response") and e.response is not None:
            try:
                error_body = e.response.json()
                error_message = 'Blocked: ' + error_body.get("error", {}).get("message", "The prompt was blocked.")
                print(Colors.color(error_message, Colors.YELLOW))
                return {"error": error_message}
            except Exception as parse_err:
                print("Unexpected error:", e)
        else:
            print("Unexpected error:", e)


def main():
    print()
    prompt = input("Enter prompt: ")
    if prompt.strip() == "/exit_client":
        exit(0)

    response = get_openai_response(prompt)
    if isinstance(response, dict):
        if "error" in response:
            return
    print(response.choices[0].message.content)


if __name__ == "__main__":
    while True:
        main()
