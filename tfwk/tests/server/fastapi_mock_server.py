"""
FastAPI Mock Server with monocle_trace_method decorators for integration testing.

This module provides a FastAPI application with various REST endpoints decorated
with @monocle_trace_method() to test trace generation in web API scenarios.
"""
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException
from monocle_apptrace.instrumentation.common.method_wrappers import (
    monocle_trace_http_route,
    monocle_trace_method,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Mock data models
class User(BaseModel):
    id: str
    name: str
    email: str
    created_at: str

class Product(BaseModel):
    id: str
    name: str
    price: float
    category: str
    in_stock: bool

class Order(BaseModel):
    id: str
    user_id: str
    product_ids: List[str]
    total_amount: float
    status: str
    created_at: str

class CreateOrderRequest(BaseModel):
    user_id: str
    product_ids: List[str]

# Mock databases
MOCK_USERS: Dict[str, User] = {}
MOCK_PRODUCTS: Dict[str, Product] = {}
MOCK_ORDERS: Dict[str, Order] = {}

# Internal helper methods that get called as part of REST operations
@monocle_trace_method(span_name="validate_user_data")
def validate_user_data(name: str, email: str) -> Dict[str, str]:
    """Validate user input data with business logic."""
    logger.debug(f"Validating user data: name='{name}', email='{email}'")
    time.sleep(0.02)  # Simulate validation processing
    
    errors = {}
    if not name or len(name.strip()) < 2:
        errors["name"] = "Name must be at least 2 characters long"
    if not email or "@" not in email:
        errors["email"] = "Invalid email format"
    
    return errors

@monocle_trace_method(span_name="calculate_user_profile_score")
def calculate_user_profile_score(user: User) -> int:
    """Calculate a profile completeness score for a user."""
    logger.debug(f"Calculating profile score for user {user.id}")
    time.sleep(0.03)  # Simulate complex calculation
    
    score = 0
    if user.name and len(user.name) > 2:
        score += 30
    if user.email and "@" in user.email:
        score += 40
    if user.created_at:
        score += 30
    
    return score

@monocle_trace_method(span_name="check_product_availability")
def check_product_availability(product_id: str) -> Dict[str, any]:
    """Check product availability and return detailed status."""
    logger.debug(f"Checking availability for product {product_id}")
    time.sleep(0.04)  # Simulate inventory check
    
    if product_id not in MOCK_PRODUCTS:
        return {"available": False, "reason": "Product not found"}
    
    product = MOCK_PRODUCTS[product_id]
    if not product.in_stock:
        return {"available": False, "reason": "Out of stock"}
    
    return {
        "available": True, 
        "stock_level": "high",  # Simulated stock level
        "estimated_delivery": "2-3 days"
    }

@monocle_trace_method(span_name="calculate_order_pricing")
def calculate_order_pricing(product_ids: List[str]) -> Dict[str, float]:
    """Calculate detailed pricing breakdown for an order."""
    logger.debug(f"Calculating pricing for products: {product_ids}")
    time.sleep(0.05)  # Simulate pricing calculation
    
    subtotal = 0.0
    for product_id in product_ids:
        if product_id in MOCK_PRODUCTS:
            subtotal += MOCK_PRODUCTS[product_id].price
    
    tax_rate = 0.08  # 8% tax
    tax_amount = subtotal * tax_rate
    shipping = 5.99 if subtotal < 50 else 0.0  # Free shipping over $50
    total = subtotal + tax_amount + shipping
    
    return {
        "subtotal": subtotal,
        "tax": tax_amount,
        "shipping": shipping,
        "total": total
    }

@monocle_trace_method(span_name="send_notification")
def send_notification(user_id: str, notification_type: str, message: str) -> bool:
    """Send a notification to a user (simulated)."""
    logger.debug(f"Sending {notification_type} notification to user {user_id}")
    time.sleep(0.08)  # Simulate external API call
    
    # Simulate notification success/failure
    if user_id in MOCK_USERS:
        logger.info(f"Notification sent: {message}")
        return True
    else:
        logger.warning(f"Failed to send notification: User {user_id} not found")
        return False

@monocle_trace_method(span_name="audit_log_operation")
def audit_log_operation(operation: str, entity_type: str, entity_id: str, user_context: Optional[str] = None) -> None:
    """Log operations for audit trail."""
    logger.debug(f"Audit logging: {operation} on {entity_type} {entity_id}")
    time.sleep(0.01)  # Simulate audit log write
    
    audit_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "operation": operation,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "user_context": user_context
    }
    logger.info(f"Audit log entry: {audit_entry}")

# Initialize some mock data
def initialize_mock_data():
    """Initialize the mock database with sample data."""

# FastAPI app
app = FastAPI(title="Mock API Server", description="FastAPI server with monocle_trace_method decorators")

# User endpoints
@app.get("/users")
@monocle_trace_http_route
def get_all_users() -> List[User]:
    """Get all users from the mock database."""
    logger.info("Fetching all users")
    # Simulate some processing time
    time.sleep(0.1)
    return list(MOCK_USERS.values())

@app.get("/users/{user_id}")
@monocle_trace_method(span_name="get_user_by_id")
def get_user_by_id(user_id: str) -> User:
    """Get a specific user by ID."""
    logger.info(f"Fetching user with ID: {user_id}")
    # Simulate database lookup time
    time.sleep(0.05)
    
    if user_id not in MOCK_USERS:
        raise HTTPException(status_code=404, detail=f"User with ID {user_id} not found")
    
    user = MOCK_USERS[user_id]
    
    # Calculate profile score as part of user retrieval
    profile_score = calculate_user_profile_score(user)
    audit_log_operation("READ", "USER", user_id)
    
    logger.info(f"Retrieved user {user_id} with profile score: {profile_score}")
    return user

@app.post("/users")
@monocle_trace_http_route
def create_user(name: str, email: str) -> User:
    """Create a new user."""
    logger.info(f"Creating user with name: {name}, email: {email}")
    
    # Validate user data using traced method
    validation_errors = validate_user_data(name, email)
    if validation_errors:
        raise HTTPException(status_code=400, detail=f"Validation errors: {validation_errors}")
    
    # Simulate database insertion
    time.sleep(0.1)
    
    user_id = str(uuid4())[:8]  # Short UUID for demo
    user = User(
        id=user_id,
        name=name,
        email=email,
        created_at=datetime.utcnow().isoformat() + "Z"
    )
    
    MOCK_USERS[user_id] = user
    
    # Calculate profile score and send notification using traced methods
    profile_score = calculate_user_profile_score(user)
    send_notification(user_id, "welcome", f"Welcome {name}! Your profile score: {profile_score}")
    audit_log_operation("CREATE", "USER", user_id, f"user:{name}")
    
    logger.info(f"User created with ID: {user_id}")
    return user

# Product endpoints
@app.get("/products")
@monocle_trace_method(span_name="get_all_products")
def get_all_products(category: Optional[str] = None) -> List[Product]:
    """Get all products, optionally filtered by category."""
    logger.info(f"Fetching products, category filter: {category}")
    time.sleep(0.08)
    
    products = list(MOCK_PRODUCTS.values())
    if category:
        products = [p for p in products if p.category.lower() == category.lower()]
    
    return products

@app.get("/products/{product_id}")
@monocle_trace_method(span_name="get_product_by_id")
def get_product_by_id(product_id: str) -> Product:
    """Get a specific product by ID."""
    logger.info(f"Fetching product with ID: {product_id}")
    time.sleep(0.03)
    
    if product_id not in MOCK_PRODUCTS:
        raise HTTPException(status_code=404, detail=f"Product with ID {product_id} not found")
    
    # Check product availability as part of retrieval
    availability = check_product_availability(product_id)
    audit_log_operation("READ", "PRODUCT", product_id)
    
    logger.info(f"Product {product_id} availability: {availability}")
    return MOCK_PRODUCTS[product_id]

@app.put("/products/{product_id}/stock")
@monocle_trace_http_route
def update_product_stock(product_id: str, in_stock: bool) -> Dict[str, str]:
    """Update product stock status."""
    logger.info(f"Updating stock for product {product_id} to {in_stock}")
    time.sleep(0.12)
    
    if product_id not in MOCK_PRODUCTS:
        raise HTTPException(status_code=404, detail=f"Product with ID {product_id} not found")
    
    MOCK_PRODUCTS[product_id].in_stock = in_stock
    return {"message": f"Product {product_id} stock updated to {in_stock}"}

# Order endpoints
@app.get("/orders")
@monocle_trace_method(span_name="get_all_orders")
def get_all_orders(user_id: Optional[str] = None) -> List[Order]:
    """Get all orders, optionally filtered by user_id."""
    logger.info(f"Fetching orders, user_id filter: {user_id}")
    time.sleep(0.1)
    
    orders = list(MOCK_ORDERS.values())
    if user_id:
        orders = [o for o in orders if o.user_id == user_id]
    
    return orders

@app.get("/orders/{order_id}")
@monocle_trace_method(span_name="get_order_by_id")
def get_order_by_id(order_id: str) -> Order:
    """Get a specific order by ID."""
    logger.info(f"Fetching order with ID: {order_id}")
    time.sleep(0.06)
    
    if order_id not in MOCK_ORDERS:
        raise HTTPException(status_code=404, detail=f"Order with ID {order_id} not found")
    
    return MOCK_ORDERS[order_id]

@app.post("/orders")
@monocle_trace_http_route
def create_order(request: CreateOrderRequest) -> Order:
    """Create a new order."""
    user_id = request.user_id
    product_ids = request.product_ids
    logger.info(f"Creating order for user {user_id} with products {product_ids}")
    
    # Validate user exists
    if user_id not in MOCK_USERS:
        raise HTTPException(status_code=400, detail=f"User with ID {user_id} not found")
    
    # Check product availability for all items
    for product_id in product_ids:
        if product_id not in MOCK_PRODUCTS:
            raise HTTPException(status_code=400, detail=f"Product with ID {product_id} not found")
        
        availability = check_product_availability(product_id)
        if not availability["available"]:
            raise HTTPException(status_code=400, detail=f"Product {product_id} not available: {availability['reason']}")
    
    # Calculate detailed pricing using traced method
    pricing = calculate_order_pricing(product_ids)
    total_amount = pricing["total"]
    
    # Simulate order processing time
    time.sleep(0.1)
    
    order_id = str(uuid4())[:8]  # Short UUID for demo
    order = Order(
        id=order_id,
        user_id=user_id,
        product_ids=product_ids,
        total_amount=total_amount,
        status="pending",
        created_at=datetime.utcnow().isoformat() + "Z"
    )
    
    MOCK_ORDERS[order_id] = order
    
    # Send order confirmation and log the operation
    send_notification(user_id, "order_confirmation", f"Order {order_id} created for ${total_amount:.2f}")
    audit_log_operation("CREATE", "ORDER", order_id, f"user:{user_id}")
    
    logger.info(f"Order created with ID: {order_id}")
    return order

# Error simulation endpoints
@app.get("/simulate-error")
@monocle_trace_method(span_name="simulate_server_error")
def simulate_server_error():
    """Simulate a server error for testing error handling."""
    logger.warning("Simulating server error")
    time.sleep(0.05)
    raise HTTPException(status_code=500, detail="Simulated internal server error")

@app.get("/simulate-timeout")
@monocle_trace_method(span_name="simulate_timeout_operation")
def simulate_timeout():
    """Simulate a slow operation that might timeout."""
    logger.info("Simulating slow operation")
    time.sleep(2.0)  # Long delay to simulate timeout
    return {"message": "Operation completed after delay"}

# Health check endpoint (not decorated for comparison)
@app.get("/health")
def health_check():
    """Health check endpoint without tracing."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# Server management
class MockAPIServer:
    """Helper class to manage the FastAPI mock server lifecycle."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8097):
        self.host = host
        self.port = port
        self.server_thread = None
        self.server = None
        
    def start(self):
        """Start the FastAPI server in a background thread."""
        if self.server_thread and self.server_thread.is_alive():
            logger.warning("Server is already running")
            return
            
        def run_server():
            config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning")
            server = uvicorn.Server(config)
            self.server = server
            server.run()
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        # Wait for server to start
        import requests
        for i in range(30):  # Wait up to 3 seconds
            try:
                response = requests.get(f"http://{self.host}:{self.port}/health", timeout=1)
                if response.status_code == 200:
                    logger.info(f"Mock API server started at http://{self.host}:{self.port}")
                    break
            except Exception:
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError("Failed to start mock API server")
    
    def stop(self):
        """Stop the FastAPI server."""
        if self.server:
            self.server.should_exit = True
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5)
        logger.info("Mock API server stopped")
    
    def get_base_url(self) -> str:
        """Get the base URL of the server."""
        return f"http://{self.host}:{self.port}"

# Global server instance
mock_server = MockAPIServer()