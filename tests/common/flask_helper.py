from flask import Flask, request, jsonify
from threading import Thread
import time
from common.chain_exec import exec_chain
PORT=8096
import multiprocessing

web_app = Flask(__name__)
flask_proc:multiprocessing.Process = None

@web_app.route("/chat", methods=["GET"])
def message():
    try:
        client_id= request.headers["client-id"]
        question = request.args["question"]
        response = exec_chain(question)
        return response
    except Exception as e:
        print(e)
        return jsonify({"Status":"Failure --- some error occured"})

def start_server():
    global web_app
    web_app.run(host="127.0.0.1", port=PORT)

def stop_flask():
    global flask_proc
    if flask_proc is not None:
        flask_proc.terminate()

def start_flask():
    global flask_proc
    flask_proc = multiprocessing.Process(target=start_server)
    flask_proc.start()
    time.sleep(2)

def get_url() -> str:
    return f"http://127.0.0.1:{PORT}"