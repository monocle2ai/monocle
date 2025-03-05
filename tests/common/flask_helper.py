from flask import Flask, request, jsonify
from threading import Thread
import time, logging
import requests
from common.chain_exec import exec_chain
PORT=8096
import multiprocessing
from monocle_apptrace.instrumentation.common.instrumentor import monocle_trace_http_route

web_app = Flask(__name__)
flask_proc:multiprocessing.Process = None

def route_executer(request):
    try:
        client_id= request.headers["client-id"]
        question = request.args["question"]
        response = exec_chain(question)
        return response
    except Exception as e:
        print(e)
        return jsonify({"Status":"Failure --- some error occured"})

                
@web_app.route("/chat", methods=["GET"])
def message_chat():
    route_executer(request)

@web_app.route("/talk", methods=["GET"])
async def talk():
    route_executer(request)

@web_app.route("/http_chat", methods=["GET"])
@monocle_trace_http_route
def message_http_chat():
    route_executer(request)

@web_app.route("/http_talk", methods=["GET"])
@monocle_trace_http_route
async def talk_http():
    route_executer(request)

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