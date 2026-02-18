from aiohttp import web
from aiohttp.test_utils import TestServer, TestClient
import aiohttp
from common.chain_exec import exec_chain
import asyncio
import logging

logger = logging.getLogger(__name__)
PORT= 8082

async def chat_handler(request):
    # Extract the 'question' parameter from the query string
    try:
        question = request.query.get('question')
        client_id = request.headers.get("client-id")
        response = exec_chain(question)
    except Exception as e:
        logger.info(e)
        response = "Failure {e}"
    return web.Response(text=response)

def health_check(request):
    return web.Response(text="")

def hello(request):
    return web.Response(text="Status:Success")

def create_app():
    app = web.Application()
    app.router.add_get('/', health_check)
    app.router.add_get('/chat', chat_handler)
    app.router.add_get('/hello', hello)
    return app

async def run_server():
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', PORT)
    await site.start()
    logger.info(f"AIOHTTP server running on http://localhost:{PORT}")
    return runner

async def stop_server(runner):
    if runner:
        await runner.cleanup()

async def check_server():
    for i in range(15):
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(get_url() + "/hello") as response:
                    if response.status == 200:
                        logger.info("AIOHTTP server started")
                        return
        except Exception as e:
            pass
        await asyncio.sleep(1)

def get_url():
    return f"http://localhost:{PORT}"