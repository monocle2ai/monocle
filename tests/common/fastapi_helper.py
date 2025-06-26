import asyncio
import uuid
import threading
import time
# from common.chain_exec import exec_chain

PORT = 8083

# Try to import FastAPI, handle gracefully if not available
try:
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available. Install with: pip install fastapi")

if FASTAPI_AVAILABLE:
    app = FastAPI()
else:
    app = None

if FASTAPI_AVAILABLE:
    @app.get("/chat")
    async def chat_handler(request: Request, question: str = None):
        """Chat endpoint handler similar to aiohttp version."""
        try:
            client_id = request.headers.get("client-id")
            if not question:
                raise HTTPException(status_code=400, detail="Question parameter is required")
            # response = exec_chain(question)
            return {"response": ""}
        except Exception as e:
            print(f"Error in chat handler: {e}")
            return JSONResponse(content={"error": f"Failure: {e}"}, status_code=500)

    @app.get("/hello")
    async def hello():
        """Health check endpoint."""
        return {"Status": "Success"}

import asyncio
import uuid
import threading
import time
# from common.chain_exec import exec_chain

PORT = 8083

# Try to import FastAPI, handle gracefully if not available
try:
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available. Install with: pip install fastapi")

if FASTAPI_AVAILABLE:
    app = FastAPI()

    @app.get("/chat")
    async def chat_handler(request: Request, question: str = None):
        """Chat endpoint handler similar to aiohttp version."""
        try:
            client_id = request.headers.get("client-id")
            if not question:
                raise HTTPException(status_code=400, detail="Question parameter is required")
            response = ""
            return {"response": response}
        except Exception as e:
            print(f"Error in chat handler: {e}")
            return JSONResponse(content={"error": f"Failure: {e}"}, status_code=500)

    @app.get("/hello")
    async def hello():
        """Health check endpoint."""
        return {"Status": "Success"}
else:
    app = None

class FastAPIServer:
    """FastAPI server wrapper for testing using threading approach."""
    
    def __init__(self):
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for FastAPI tests")
        self.server = None
        self.server_thread = None
        self.running = False
        
    async def start(self):
        """Start the FastAPI server."""
        try:
            # Try using uvicorn if available
            import uvicorn
            
            def run_server():
                uvicorn.run(app, host="localhost", port=PORT, log_level="error")
                
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            
        except ImportError:
            # Simple fallback - just mark as running for testing
            print("Uvicorn not available, running in mock mode")
            self.running = True
            
        # Wait for server to be ready
        await self.wait_for_server()
        print(f"FastAPI server running on http://localhost:{PORT}")
        return self
        
    async def wait_for_server(self):
        """Wait for server to be ready."""
        if not hasattr(self, 'server_thread') or not self.server_thread:
            # Mock mode - just wait a bit
            await asyncio.sleep(0.5)
            self.running = True
            return
            
        import aiohttp
        for i in range(15):
            try:
                timeout = aiohttp.ClientTimeout(total=5)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(f"http://localhost:{PORT}/hello?abc=1") as response:
                        if response.status == 200:
                            print("FastAPI server started")
                            self.running = True
                            return
            except Exception:
                pass
            await asyncio.sleep(1)
        
        print("Could not verify server start, continuing anyway")
        self.running = True
        
    async def cleanup(self):
        """Stop the FastAPI server."""
        self.running = False
        # Note: In a real scenario, we'd need a more sophisticated shutdown mechanism
        # For testing purposes, the daemon thread will be cleaned up automatically

async def run_server():
    """Run the FastAPI server for testing."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required for FastAPI tests")
    server = FastAPIServer()
    await server.start()
    return server

async def stop_server(server):
    """Stop the FastAPI server."""
    if server:
        await server.cleanup()

def get_url():
    """Get the server URL."""
    return f"http://localhost:{PORT}"
