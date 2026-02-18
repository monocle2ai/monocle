from flask import Flask, request, jsonify
from threading import Thread
import time, logging
import requests
from common.chain_exec import exec_chain
import logging

logger = logging.getLogger(__name__)
PORT=8096

web_app = Flask(__name__)

def route_executer(request):
    try:
        client_id= request.headers["client-id"]
        question = request.args["question"]
        response = exec_chain(question)
        return response
    except Exception as e:
        logger.info(e)
        return jsonify({"Status":"Failure --- some error occured"})

                
@web_app.route("/chat", methods=["GET"])
def message_chat():
    return route_executer(request)

@web_app.route("/", methods=["GET"])
def health_check():
    return jsonify({})

@web_app.route("/hello", methods=["GET"])
def hello():
    return jsonify({"Status":"Success"})

def start_server() -> None:
    global web_app
    web_app.run(host="127.0.0.1", port=PORT)

def stop_flask():
    pass

def start_flask():
    logger.info("Going to start Flask server")
    flask_thread = Thread(target=lambda: start_server())
    flask_thread.daemon = True
    flask_thread.start()
    for i in range(15):
        try:
            response = requests.get(get_url()+"/hello")
            if response.status_code == 200:
                logger.info("Flask server started")
                break
        except Exception as e:
            pass
        time.sleep(1)

def get_url() -> str:
    return f"http://127.0.0.1:{PORT}"