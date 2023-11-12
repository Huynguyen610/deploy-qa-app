import torch
import chainlit as cl
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


def make_prediction(context, question):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModelForQuestionAnswering.from_pretrained("huynguyen61098/Bert-Base-Cased-Squad-Extractive-QA").to(device)
    inputs = tokenizer(question, context, return_tensors="pt")
    print("Context:")
    print(context)
    print("Question:")
    print(question)
    outputs = model(**inputs)
    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()
    predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
    answers = tokenizer.decode(predict_answer_tokens)
    print("\nModel Answer:")
    print(answers)
    return answers


@cl.on_chat_start
async def start():
    res = await cl.AskUserMessage(content="What is your name?", timeout=30).send()
    if res:
        await cl.Message(
            content=f"Hi {res['content']}.\nI am an Extractive Question-Answering Bot!\nPlease type \'next\' to continue"
        ).send()


@cl.on_message
async def main(message: cl.Message):
    # ask for user input
    context_res = await cl.AskUserMessage(
        content="Please provide your Text (context for your question: paragraph, sentence,etc..)!",
        timeout=60).send()

    if context_res:
        await cl.Message(
            content=f'Your context is:\n{context_res["content"]}.'
        ).send()
    user_context = context_res["content"]

    question_res = await cl.AskUserMessage(content="Now provide your question.", timeout=60,
                                           raise_on_timeout=True).send()

    if question_res:

        await cl.Message(
            content=f'Your question is:\n{question_res["content"]}.'
        ).send()
    user_question = question_res["content"]

    # Your custom logic go here
    msg = cl.Message(content="Please wait for me to process your request!")
    await msg.send()
    answer = make_prediction(context=str(user_context), question=str(user_question))
    await cl.Message(content=answer).send()


@cl.on_chat_end
async def end():
    msg = cl.Message(content="Goodbye. Thank you for visiting me!\nReload the page to start a new session.")
    await msg.send()