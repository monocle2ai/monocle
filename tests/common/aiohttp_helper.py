from aiohttp import web
from aiohttp.test_utils import TestServer, TestClient

from common.chain_exec import exec_chain
from threading import Thread
import requests
import time

PORT= 8082

async def chat_handler(request):
    # Extract the 'question' parameter from the query string
    try:
        question = request.query.get('question')
        client_id = request.headers.get("client-id")
        response = exec_chain(question)
    except Exception as e:
        print(e)
        response = "Failure {e}"
    return web.Response(text=response)

def hello():
    return web.Response(text="Status:Success")

def create_app():
    app = web.Application()
    app.router.add_get('/chat', chat_handler)
    app.router.add_get('/hello', hello)
    return app

async def run_server():
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', PORT)
    await site.start()
    return runner

async def stop_server(runner):
    if runner:
        await runner.cleanup()

async def check_server():
    for i in range(15):
        try:
            response = requests.get(get_url()+"/hello")
            if response.status_code == 200:
                print("Flask server started")
                break
        except Exception as e:
            pass
        time.sleep(1)

def get_url():
    return f"http://localhost:{PORT}"

if __name__ == '__main__':
    run_server()


