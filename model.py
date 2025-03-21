from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import logging
import time
from transformers import AutoTokenizer

# Configure logging to a file
logging.basicConfig(filename='chatbot.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_FAISS_PATH = 'vectorstore3/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2")


def truncate_input(input_text, max_length=512):
    tokens = tokenizer.encode(input_text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    return tokenizer.decode(tokens)


def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=[
                            'context', 'question'])
    return prompt


def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain


def load_llm():
    llm = CTransformers(
        model="models/TheBloke_Llama-2-7B-Chat-GGML/llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm


def qa_bot():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        db = FAISS.load_local(DB_FAISS_PATH, embeddings,
                              allow_dangerous_deserialization=True)
        llm = load_llm()
        qa_prompt = set_custom_prompt()
        qa = retrieval_qa_chain(llm, qa_prompt, db)
        return qa
    except Exception as e:
        logger.error("Error setting up QA bot: %s", e)
        return None


def final_result(query):
    truncated_query = truncate_input(query, max_length=512)
    start_time = time.time()  # Start time measurement

    qa = qa_bot()
    if qa:
        response = qa({'query': truncated_query})
    else:
        response = {"result": "Error setting up the QA bot"}

    end_time = time.time()  # End time measurement
    elapsed_time = end_time - start_time

    logger.info(f"Response time: {elapsed_time} seconds")
    return response


@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Sanjeevini chatBot."
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    if not chain:
        await cl.Message(content="Error: QA bot not initialized.").send()
        return

    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    start_time = time.time()

    try:
        await cl.Message(content="Processing your query, please wait...").send()
        res = await chain.acall(message.content, callbacks=[cb])
        answer = res["result"]

        # Send the response
        await cl.Message(content=answer).send()
    except Exception as e:
        logger.error("Error processing message: %s", e)
        await cl.Message(content="Error processing your query. Please try again later.").send()

    end_time = time.time()
    elapsed_time = end_time - start_time

    logger.info(
        f"Response time for query '{message.content}': {elapsed_time} seconds")
