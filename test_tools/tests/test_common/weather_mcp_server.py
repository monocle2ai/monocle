import os
import contextlib
import random
import uvicorn
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from mcp.server.fastmcp import FastMCP

DEFAULT_PORT = 8001
port = int(os.getenv("PORT", DEFAULT_PORT))

# Initialize the weather MCP server
weather_mcp = FastMCP(name="WeatherServer", stateless_http=True)

@weather_mcp.tool()
async def demo_get_weather(city: str) -> Dict[str, Any]:
    """
    Get mock weather information for a specified city.
    
    Args:
        city: Name of the city to get weather for (e.g., "London", "New York", "Tokyo")
    
    Returns:
        Dictionary containing mock weather information with random temperature between 40-100.
    """
    if not city.strip():
        raise HTTPException(status_code=400, detail="City name cannot be empty")

    
    # Generate random temperature between 40 and 100
    temperature = random.randint(40, 100)
    
    # Return simple mock response
    return {
        "temperature": temperature
    }

# Create a combined lifespan to manage all session managers
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(weather_mcp.session_manager.run())
        yield

# Initialize FastAPI app
app = FastAPI(
    lifespan=lifespan,
    title="Weather MCP Server",
    description="FastAPI MCP server weather service",
    version="1.0.0"
)

app.mount("/weather", weather_mcp.streamable_http_app())

# allow cors for all origins
@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response



if __name__ == "__main__":
    print("Starting Weather MCP Server ...")
    print(f"Server will be available at: http://localhost:{port}")
    print("Returns mock weather data with random temperature between 40-100")
    
    uvicorn.run(
        "weather_mcp_server:app",
        host="127.0.0.1",
        port=port,
        reload=True,
        log_level="info"
    )
