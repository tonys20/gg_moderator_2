import os
import logging
from dotenv import load_dotenv
from misc import Colors
import httpx

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

def get_openai_response(prompt):
    client = httpx.Client(verify=False, timeout=60)

    response = client.post(
        os.getenv("OPENAI_BASE_URL"),
        headers={"Authorization": f"Bearer {os.getenv('OPENAI_KEY')}"},
        json={
            "model": os.getenv("OPENAI_MODEL"),
            "messages": [{"role": "user", "content": prompt}]
        }
    )

    return response.json()

def main():
    prompt = input("Enter prompt: ")
    if prompt.strip() == "/exit_client":
        exit(0)

    response = get_openai_response(prompt)
    if isinstance(response, dict):
        if "error" in response:
            print(Colors.YELLOW + response["error"]['message'] + Colors.RESET)
            return
    print(response['choices'][0]['message']['content'])


if __name__ == "__main__":
    while True:
        main()
