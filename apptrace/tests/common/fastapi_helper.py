import logging
import multiprocessing
import time
from threading import Thread

import requests
import uvicorn
from common.chain_exec import exec_chain
from fastapi import FastAPI, Request

PORT = 8096
app = FastAPI()
server_process = None

logger = logging.getLogger(__name__)

def route_executer(request: Request):
    try:
        client_id = request.headers["client-id"]
        question = request.query_params["question"]
        response = exec_chain(question)
        return response
    except Exception as e:
        logger.error(f"Error in route execution: {e}")
        return {"Status": "Failure --- some error occurred"}

@app.get("/chat")
def message_chat(request: Request):
    return route_executer(request)

@app.get("/hello")
def hello():
    return {"Status": "Success"}

def start_server():
    uvicorn.run(app, host="127.0.0.1", port=PORT)

def stop_fastapi():
    global server_process
    if server_process:
        server_process = None
        logger.info("FastAPI server stopped")

def start_fastapi():
    global server_process
    logger.info("Starting FastAPI server")

    # Clean up any existing server process
    #stop_fastapi()
    server_process = Thread(target=lambda: start_server())
    server_process.daemon = True
    server_process.start()
    # Start new server process
    # server_process = multiprocessing.Process(target=start_server)
    # server_process.daemon = True
    # server_process.start()
    for i in range(15):
        try:
            response = requests.get(get_url() + "/hello")
            if response.status_code == 200:
                logger.info("FastAPI server started successfully")
                break
        except Exception:
            pass
        time.sleep(1)

def get_url() -> str:
    return f"http://127.0.0.1:{PORT}"