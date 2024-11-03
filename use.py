import requests

IP = "IP"
PORT = "PORT"
# BASE_URL = "http://DOMAIN"
BASE_URL = f"http://{IP}:{PORT}"

def get_ai_answer(question: str, max_length: int, max_new_tokens: int):
    params = {
        'max_length': max_length,
        'max_new_tokens': max_new_tokens,
    }

    response = requests.post(f'{BASE_URL}/llama', params=params, json=question, verify=False)

    if response.status_code != 200:
        raise Exception(response.text)

    return response.json()

def get_ai_answer_from_context(context: str, question: str):
    params = {
        'question': question,
    }

    response = requests.post(f'{BASE_URL}/context', params=params, json=context, verify=False)

    if response.status_code != 200:
        raise Exception(response.text)

    return response.json()