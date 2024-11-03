import gc

import torch
import uvicorn
from fastapi import FastAPI, Body
from transformers import pipeline

app = FastAPI(
    title="AI API"
)

pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-3B-Instruct",
                device_map="auto", torch_dtype=torch.bfloat16)

model_name = "timpal0l/mdeberta-v3-base-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name, device='cuda:0')


async def get_ai_answer(question: str, max_length: int, max_new_tokens: int):
    response = None
    try:
        messages = [
            {"role": "system", "content": "Ты ассистент который должен отвечать на задаваемые вопросы!"},
            {"role": "user", "content": question},
        ]

        response = pipe(messages, max_length=max_length, max_new_tokens=max_new_tokens)
        answer = response[0]['generated_text'][-1]['content']
        response.clear()
        response = None

        return answer
    except Exception as e:
        raise e
    finally:
        gc.collect()

    return response


async def get_ai_answer_from_context(context: str, question: str):
    QA_input = {
        'question': question,
        'context': context
    }
    res = nlp(QA_input)

    return res


@app.post("/llama")
async def ai_answer(max_length: int, max_new_tokens: int, question: str = Body(...)):
    return get_ai_answer(question, max_length, max_new_tokens)

@app.post("/context")
async def ai_answer_from_context(question: str, context: str = Body(...)):
    return get_ai_answer_from_context(context, question)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8998)
