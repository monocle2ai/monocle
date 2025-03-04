from flask import Flask, request, jsonify
from threading import Thread
import time, logging
import requests
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

@web_app.route("/hello", methods=["GET"])
def hello():
    return jsonify({"Status":"Success"})

def start_server():
    global web_app
    web_app.run(host="127.0.0.1", port=PORT)

def stop_flask():
    pass

def start_flask():
    flask_thread = Thread(target=start_server)
    flask_thread.daemon = True
    flask_thread.start()
    for i in range(10):
        try:
            requests.get(get_url()+"/hello")
            break
        except:
            time.sleep(1)

def get_url() -> str:
    return f"http://127.0.0.1:{PORT}"