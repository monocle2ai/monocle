from fastapi import FastAPI, Request
from threading import Thread
import uvicorn
import time
import requests
from common.chain_exec import exec_chain

PORT = 8096
app = FastAPI()

def route_executer(request: Request):
    try:
        client_id = request.headers["client-id"]
        question = request.query_params["question"]
        response = exec_chain(question)
        return response
    except Exception as e:
        print(e)
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
    pass

def start_fastapi():
    print("Going to start FastAPI server")
    fastapi_thread = Thread(target=start_server)
    fastapi_thread.daemon = True
    fastapi_thread.start()
    for i in range(15):
        try:
            response = requests.get(get_url() + "/hello")
            if response.status_code == 200:
                print("FastAPI server started")
                break
        except Exception:
            pass
        time.sleep(1)

def get_url() -> str:
    return f"http://127.0.0.1:{PORT}"