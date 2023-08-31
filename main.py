import time

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from typing import Optional
from dotenv import load_dotenv
import logging

from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain, \
    ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import VectorStore

from babyhelpergpt.agents import BabyChatGPT
from schemas import ChatResponse
templates = Jinja2Templates(directory="templates")
# vectorstore: Optional[VectorStore] = None
from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
load_dotenv(dotenv_path='.env')
app = FastAPI()

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()
    stream_handler = StreamingLLMCallbackHandler(websocket)

    streaming_llm = ChatOpenAI(
        # model="gpt-3.5-turbo",
        model_name = 'gpt-3.5-turbo-0613',
        streaming=True,
        callbacks=[stream_handler],
        verbose=True,
        temperature=0.2,
    )
    llm_analizer = ChatOpenAI(
        verbose=True,
        temperature=0.0,
    )
    helper_agent = BabyChatGPT.from_llm_my(llm=streaming_llm, llm_analizer=llm_analizer, verbose=False)

    resp = ChatResponse(sender="bot", message='', type="start")
    await websocket.send_json(resp.dict())
    helper_agent.seed_agent()
    while True:
        try:

            helper_agent.determine_conversation_stage()
            await helper_agent.astep()
            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")

            await websocket.send_json(resp.dict())
            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())


            #end conversation
            if "<END_OF_CALL>" in helper_agent.conversation_history[-1]:
                print("Sales Agent determined it is time to end the conversation.")
                break
            helper_agent.human_step(question)

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
            # helper_agent.determine_conversation_stage()
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)