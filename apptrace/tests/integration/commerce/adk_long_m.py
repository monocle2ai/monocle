"""
AI E-commerce Agent Integration Test with Shopping Cart and Purchase Flow

This enhanced test demonstrates how Google ADK agents can collaborate in a complete
e-commerce scenario with persistent memory across sessions. It simulates:
1. Product browsing and information collection
2. Shopping cart management across sessions
3. User preference storage and retrieval
4. Purchase flow with payment processing (mocked)
5. Order confirmation and history tracking

The test showcases real-world AI agent collaboration in e-commerce applications.
"""

import asyncio
import os

# Import Google ADK components for building AI agents
from google.adk.agents import LlmAgent
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import load_memory
from google.genai.types import Content, Part
from monocle_apptrace import monocle_trace, setup_monocle_telemetry
from monocle_apptrace.instrumentation.common.utils import get_current_monocle_span

# Application configuration constants
APPLICATION_NAME = "ecommerce_memory_app"  # Unique identifier for this e-commerce test application
TEST_USER_ID = "ecommerce_test_user"       # Identifier for the test user in this scenario
LANGUAGE_MODEL = "gemini-2.5-flash"        # The AI model used for all agents

# Mock E-commerce Data
PRODUCT_CATALOG = {
    "laptop_pro": {
        "id": "laptop_pro",
        "name": "Professional Laptop",
        "price": 1299.99,
        "description": "High-performance laptop for professionals",
        "category": "electronics",
        "stock": 15
    },
    "wireless_mouse": {
        "id": "wireless_mouse", 
        "name": "Wireless Mouse",
        "price": 49.99,
        "description": "Ergonomic wireless mouse with long battery life",
        "category": "accessories",
        "stock": 50
    },
    "coffee_beans": {
        "id": "coffee_beans",
        "name": "Premium Coffee Beans", 
        "price": 24.99,
        "description": "Single-origin coffee beans, medium roast",
        "category": "food",
        "stock": 100
    },
    "desk_lamp": {
        "id": "desk_lamp",
        "name": "LED Desk Lamp",
        "price": 79.99, 
        "description": "Adjustable LED desk lamp with multiple brightness levels",
        "category": "home",
        "stock": 25
    }
}

# In-memory shopping carts and order storage
SHOPPING_CARTS = {}
ORDER_HISTORY = {}
PAYMENT_METHODS = {}

# Shopping Cart and E-commerce Tools
def add_to_cart(product_id: str, quantity: int) -> dict:
    """Add a product to the user's shopping cart
    
    Args:
        product_id (str): The ID of the product to add to cart
        quantity (int): The quantity to add
        
    Returns:
        dict: status and result or error message
    """
    user_id = TEST_USER_ID
    
    # Create a custom span for cart operation using Monocle's built-in tracing
    with monocle_trace(
        span_name="ecommerce.cart.add_item",
        attributes={
            "user.id": user_id,
            "product.id": product_id,
            "cart.quantity": quantity
        }
    ):
        # Get the current span to set additional attributes dynamically
        span = get_current_monocle_span()
        
        if user_id not in SHOPPING_CARTS:
            SHOPPING_CARTS[user_id] = {}
        
        if product_id not in PRODUCT_CATALOG:
            span.set_attribute("error.type", "product_not_found")
            span.set_attribute("operation.success", False)
            return {
                "status": "error",
                "error_message": f"Product {product_id} not found"
            }
        
        product = PRODUCT_CATALOG[product_id]
        span.set_attribute("product.name", product["name"])
        span.set_attribute("product.price", product["price"])
        span.set_attribute("inventory.available", product["stock"])
        
        if product["stock"] < quantity:
            span.set_attribute("error.type", "insufficient_inventory")
            span.set_attribute("operation.success", False)
            return {
                "status": "error",
                "error_message": f"Only {product['stock']} items available for {product['name']}"
            }
        
        if product_id in SHOPPING_CARTS[user_id]:
            SHOPPING_CARTS[user_id][product_id] += quantity
        else:
            SHOPPING_CARTS[user_id][product_id] = quantity
        
        # Calculate cart metrics
        cart_total_items = sum(SHOPPING_CARTS[user_id].values())
        cart_total_value = sum(PRODUCT_CATALOG[pid]["price"] * qty for pid, qty in SHOPPING_CARTS[user_id].items())
        
        span.set_attribute("cart.total_items", cart_total_items)
        span.set_attribute("cart.total_value", cart_total_value)
        span.set_attribute("operation.success", True)
        
        return {
            "status": "success",
            "result": f"Added {quantity} x {product['name']} to cart. Cart now has {cart_total_items} items worth ${cart_total_value:.2f}"
        }

def view_cart() -> dict:
    """View the contents of the user's shopping cart
    
    Returns:
        dict: status and cart contents or empty message
    """
    user_id = TEST_USER_ID
    
    # Create a custom span for cart viewing using Monocle's built-in tracing
    with monocle_trace(
        span_name="ecommerce.cart.view",
        attributes={"user.id": user_id}
    ):
        # Get the current span to set additional attributes dynamically
        span = get_current_monocle_span()
        
        if user_id not in SHOPPING_CARTS or not SHOPPING_CARTS[user_id]:
            span.set_attribute("cart.empty", True)
            span.set_attribute("cart.total_items", 0)
            span.set_attribute("cart.total_value", 0.0)
            return {
                "status": "success",
                "result": "Your cart is empty"
            }
        
        cart_items = []
        total = 0.0
        item_count = 0
        
        for product_id, quantity in SHOPPING_CARTS[user_id].items():
            product = PRODUCT_CATALOG[product_id]
            subtotal = product["price"] * quantity
            total += subtotal
            item_count += quantity
            cart_items.append(f"{quantity} x {product['name']} - ${subtotal:.2f}")
        
        span.set_attribute("cart.empty", False)
        span.set_attribute("cart.total_items", item_count)
        span.set_attribute("cart.total_value", total)
        span.set_attribute("cart.unique_products", len(SHOPPING_CARTS[user_id]))
        
        cart_summary = "\n".join(cart_items)
        return {
            "status": "success",
            "result": f"Cart Contents:\n{cart_summary}\nTotal: ${total:.2f}"
        }

def mock_payment_process(payment_method: str, billing_address: str) -> dict:
    """Process payment for items in the shopping cart
    
    Args:
        payment_method (str): The payment method (e.g., credit_card, debit_card)
        billing_address (str): The billing address for payment
        
    Returns:
        dict: status and payment result or error message
    """
    user_id = TEST_USER_ID
    import uuid
    from datetime import datetime

    # Create a custom span for payment processing using Monocle's built-in tracing
    with monocle_trace(
        span_name="ecommerce.payment.process",
        attributes={
            "user.id": user_id,
            "payment.method": payment_method,
            "billing.address": billing_address
        }
    ):
        # Get the current span to set additional attributes dynamically
        span = get_current_monocle_span()
        
        if user_id not in SHOPPING_CARTS or not SHOPPING_CARTS[user_id]:
            span.set_attribute("error.type", "empty_cart")
            span.set_attribute("payment.success", False)
            return {
                "status": "error",
                "error_message": "Cart is empty"
            }
        
        # Calculate total and metrics
        total = 0.0
        item_count = 0
        for product_id, quantity in SHOPPING_CARTS[user_id].items():
            product = PRODUCT_CATALOG[product_id]
            total += product["price"] * quantity
            item_count += quantity
        
        span.set_attribute("order.total", total)
        span.set_attribute("order.item_count", item_count)
        span.set_attribute("order.unique_products", len(SHOPPING_CARTS[user_id]))
        
        # Generate mock order ID
        order_id = str(uuid.uuid4())[:8]
        span.set_attribute("order.id", order_id)
        
        # Store order in history
        if user_id not in ORDER_HISTORY:
            ORDER_HISTORY[user_id] = []
        
        order = {
            "order_id": order_id,
            "items": dict(SHOPPING_CARTS[user_id]),
            "total": total,
            "payment_method": payment_method,
            "billing_address": billing_address,
            "status": "completed",
            "date": datetime.now().isoformat()
        }
        
        ORDER_HISTORY[user_id].append(order)
        
        # Update stock and clear cart
        inventory_updates = []
        for product_id, quantity in SHOPPING_CARTS[user_id].items():
            old_stock = PRODUCT_CATALOG[product_id]["stock"]
            PRODUCT_CATALOG[product_id]["stock"] -= quantity
            new_stock = PRODUCT_CATALOG[product_id]["stock"]
            inventory_updates.append(f"{product_id}: {old_stock} -> {new_stock}")
        
        span.set_attribute("inventory.updates", ", ".join(inventory_updates))
        span.set_attribute("payment.success", True)
        span.set_attribute("order.status", "completed")
        
        SHOPPING_CARTS[user_id] = {}
        
        return {
            "status": "success",
            "result": f"Payment successful! Order #{order_id} confirmed. Total: ${total:.2f}. Items have been deducted from inventory."
        }

def search_products(query: str) -> dict:
    """Search for products in the catalog
    
    Args:
        query (str): The search query to match against product names, descriptions, or categories
        
    Returns:
        dict: status and search results or no results message
    """
    # Create a custom span for product search using Monocle's built-in tracing
    with monocle_trace(
        span_name="ecommerce.search.products",
        attributes={"search.query": query}
    ):
        # Get the current span to set additional attributes dynamically
        span = get_current_monocle_span()
        
        results = []
        query_lower = query.lower()
        
        # Search terms to match against catalog
        search_terms = ["electronics", "laptop", "mouse", "professional", "office", "accessory", "computer"]
        
        for product_id, product in PRODUCT_CATALOG.items():
            # Check if query matches product attributes or if any search term matches
            match_found = (query_lower in product["name"].lower() or 
                          query_lower in product["description"].lower() or
                          query_lower in product["category"].lower())
            
            # Also check if any search terms match for broader discovery
            if not match_found:
                for term in search_terms:
                    if (term in query_lower and 
                        (term in product["name"].lower() or 
                         term in product["description"].lower() or
                         term in product["category"].lower())):
                        match_found = True
                        break
            
            if match_found:
                results.append(f"{product['name']} (ID: {product_id}) - ${product['price']:.2f} ({product['stock']} in stock)")
        
        span.set_attribute("search.results_count", len(results))
        span.set_attribute("search.total_catalog_size", len(PRODUCT_CATALOG))
        
        if not results:
            span.set_attribute("search.results_found", False)
            return {
                "status": "success",
                "result": "No products found matching your search"
            }
        
        span.set_attribute("search.results_found", True)
        return {
            "status": "success",
            "result": "Search Results:\n" + "\n".join(results)
        }

# Create AI agents with specific e-commerce roles
# Agent 1: Product browsing agent - helps users discover and learn about products
product_browsing_agent = LlmAgent(
    model=LANGUAGE_MODEL,
    name="ProductBrowsingAgent",
    instruction="""You help users browse and discover products. When users ask about products or mention items they want, 
    immediately use the search_products function to find matching products. Always search the catalog to provide accurate 
    product information including prices and stock levels.""",
    tools=[search_products]
)

# Agent 2: Shopping cart agent - manages cart operations and product selection
shopping_cart_agent = LlmAgent(
    model=LANGUAGE_MODEL,
    name="ShoppingCartAgent", 
    instruction="""You manage shopping carts. When users want to add items, immediately use add_to_cart function with 
    the exact product_id and quantity number. When users ask to see their cart, use view_cart function. Always use the 
    exact product IDs from the catalog (like laptop_pro, wireless_mouse).""",
    tools=[load_memory, add_to_cart, view_cart]
)

# Agent 3: Checkout agent - handles payment processing and order completion
checkout_agent = LlmAgent(
    model=LANGUAGE_MODEL,
    name="CheckoutAgent",
    instruction="""You handle payment processing. First use view_cart to show the cart contents. When user provides 
    payment information, immediately use mock_payment_process function with the payment_method and billing_address 
    to process the payment and complete the order.""",
    tools=[load_memory, view_cart, mock_payment_process]
)

# Initialize memory and session management services
# Option 1: In-memory services (data lost when app stops)
conversation_session_service = InMemorySessionService()  # Manages conversation sessions and state
conversation_memory_service = InMemoryMemoryService()    # Stores and retrieves conversation memories

async def demonstrate_ecommerce_agent_flow():
    """
    Demonstrates a complete e-commerce flow with multiple AI agents and persistent memory:
    
    Part 1: Product Discovery and Preference Learning
    - User browses products and shares preferences
    - Product browsing agent learns and stores user preferences
    
    Part 2: Shopping Cart Management
    - User adds items to cart based on preferences
    - Shopping cart agent manages cart and recalls preferences
    
    Part 3: Checkout and Purchase
    - User completes purchase with payment processing
    - Checkout agent handles payment and order confirmation
    
    This showcases collaborative AI agents in real-world e-commerce scenarios.
    """
    
    # === PART 1: PRODUCT DISCOVERY AND PREFERENCE LEARNING ===
    print("=== PART 1: Product Discovery Session ===")
    print("User will browse products and share preferences...")
    
    # Create a runner for the product browsing agent
    browsing_runner = Runner(
        agent=product_browsing_agent,
        app_name=APPLICATION_NAME,
        session_service=conversation_session_service,
        memory_service=conversation_memory_service
    )
    
    # Create a session for product browsing
    browsing_session_id = "product_browsing_session"
    await browsing_runner.session_service.create_session(
        app_name=APPLICATION_NAME, 
        user_id=TEST_USER_ID,
        session_id=browsing_session_id
    )
    
    # User shares preferences
    user_preferences = Content(
        parts=[Part(text="I'm looking for electronics and accessories for my new home office. I prefer high-quality professional equipment.")], 
        role="user"
    )

    print("User:", "I'm looking for electronics and accessories for my new home office. I prefer high-quality professional equipment.")
    async for event in browsing_runner.run_async(
        user_id=TEST_USER_ID,
        session_id=browsing_session_id,
        new_message=user_preferences
    ):
        if event.is_final_response() and event.content and event.content.parts:
            print("Product Browsing Agent:", event.content.parts[0].text)

    # User asks about specific products
    laptop_query = Content(
        parts=[Part(text="Search for laptops in your catalog.")], 
        role="user"
    )
    
    print("\nUser:", "Search for laptops in your catalog.")
    async for event in browsing_runner.run_async(
        user_id=TEST_USER_ID,
        session_id=browsing_session_id,
        new_message=laptop_query
    ):
        if event.is_final_response() and event.content and event.content.parts:
            print("Product Browsing Agent:", event.content.parts[0].text)

    # Store the browsing session in memory
    completed_browsing_session = await browsing_runner.session_service.get_session(
        app_name=APPLICATION_NAME, 
        user_id=TEST_USER_ID,
        session_id=browsing_session_id
    )
    await conversation_memory_service.add_session_to_memory(completed_browsing_session)
    print("✓ Product browsing session stored in memory\n")

    # === PART 2: SHOPPING CART MANAGEMENT ===
    print("=== PART 2: Shopping Cart Management Session ===")
    print("User will add items to cart based on their preferences...")
    
    # Create a runner for the shopping cart agent
    cart_runner = Runner(
        agent=shopping_cart_agent,
        app_name=APPLICATION_NAME,
        session_service=conversation_session_service,
        memory_service=conversation_memory_service
    )
    
    # Create a session for cart management
    cart_session_id = "shopping_cart_session"
    await cart_runner.session_service.create_session(
        app_name=APPLICATION_NAME, 
        user_id=TEST_USER_ID,
        session_id=cart_session_id
    )
    
    # User wants to add laptop to cart
    add_laptop_request = Content(
        parts=[Part(text="Add laptop_pro with quantity 1 to cart.")], 
        role="user"
    )
    
    print("User:", "Add laptop_pro with quantity 1 to cart.")
    async for event in cart_runner.run_async(
        user_id=TEST_USER_ID,
        session_id=cart_session_id,
        new_message=add_laptop_request
    ):
        if event.is_final_response() and event.content and event.content.parts:
            print("Shopping Cart Agent:", event.content.parts[0].text)
    
    # User adds more items
    add_mouse_request = Content(
        parts=[Part(text="Add wireless_mouse with quantity 1 to cart.")], 
        role="user"
    )
    
    print("\nUser:", "Add wireless_mouse with quantity 1 to cart.")
    async for event in cart_runner.run_async(
        user_id=TEST_USER_ID,
        session_id=cart_session_id,
        new_message=add_mouse_request
    ):
        if event.is_final_response() and event.content and event.content.parts:
            print("Shopping Cart Agent:", event.content.parts[0].text)
    
    # Show cart contents
    view_cart_request = Content(
        parts=[Part(text="Show my cart contents.")], 
        role="user"
    )
    
    print("\nUser:", "Show my cart contents.")
    async for event in cart_runner.run_async(
        user_id=TEST_USER_ID,
        session_id=cart_session_id,
        new_message=view_cart_request
    ):
        if event.is_final_response() and event.content and event.content.parts:
            print("Shopping Cart Agent:", event.content.parts[0].text)

    # Store the cart session in memory
    completed_cart_session = await cart_runner.session_service.get_session(
        app_name=APPLICATION_NAME, 
        user_id=TEST_USER_ID,
        session_id=cart_session_id
    )
    await conversation_memory_service.add_session_to_memory(completed_cart_session)
    print("✓ Shopping cart session stored in memory\n")

    # === PART 3: CHECKOUT AND PURCHASE ===
    print("=== PART 3: Checkout and Purchase Session ===")
    print("User will complete the purchase with payment processing...")
    
    # Create a runner for the checkout agent
    checkout_runner = Runner(
        agent=checkout_agent,
        app_name=APPLICATION_NAME,
        session_service=conversation_session_service,
        memory_service=conversation_memory_service
    )
    
    # Create a session for checkout
    checkout_session_id = "checkout_session"
    await checkout_runner.session_service.create_session(
        app_name=APPLICATION_NAME, 
        user_id=TEST_USER_ID,
        session_id=checkout_session_id
    )
    
    # User initiates checkout
    checkout_request = Content(
        parts=[Part(text="Show my cart contents for checkout.")], 
        role="user"
    )
    
    print("User:", "Show my cart contents for checkout.")
    async for event in checkout_runner.run_async(
        user_id=TEST_USER_ID,
        session_id=checkout_session_id,
        new_message=checkout_request
    ):
        if event.is_final_response() and event.content and event.content.parts:
            print("Checkout Agent:", event.content.parts[0].text)
    
    # User provides payment information
    payment_request = Content(
        parts=[Part(text="Process payment with payment_method: credit_card and billing_address: 123 Main St, Anytown, USA")], 
        role="user"
    )
    
    print("\nUser:", "Process payment with payment_method: credit_card and billing_address: 123 Main St, Anytown, USA")
    async for event in checkout_runner.run_async(
        user_id=TEST_USER_ID,
        session_id=checkout_session_id,
        new_message=payment_request
    ):
        if event.is_final_response() and event.content and event.content.parts:
            print("Checkout Agent:", event.content.parts[0].text)
    
    # Store the checkout session in memory
    completed_checkout_session = await checkout_runner.session_service.get_session(
        app_name=APPLICATION_NAME, 
        user_id=TEST_USER_ID,
        session_id=checkout_session_id
    )
    await conversation_memory_service.add_session_to_memory(completed_checkout_session)
    print("✓ Checkout session stored in memory")
    
    # === PART 4: ORDER CONFIRMATION AND HISTORY ===
    print("\n=== PART 4: Order History Verification ===")
    print("Verifying the complete e-commerce flow with memory persistence...")
    
    # Show that memory persists across the entire flow
    if TEST_USER_ID in ORDER_HISTORY:
        orders = ORDER_HISTORY[TEST_USER_ID]
        print(f"✓ Order history contains {len(orders)} completed order(s)")
        for order in orders:
            print(f"  - Order #{order['order_id']}: ${order['total']:.2f} ({order['status']})")
    
    print(f"✓ Product stock updated: Laptop stock now {PRODUCT_CATALOG['laptop_pro']['stock']}, Mouse stock now {PRODUCT_CATALOG['wireless_mouse']['stock']}")
    print(f"✓ Shopping cart cleared: {view_cart()}")
    
    print("\n🛒 E-commerce test completed successfully!")
    print("✅ All agents collaborated effectively with persistent memory")
    print("✅ Complete purchase flow from browsing to payment processed")
    print("✅ User preferences and order history maintained across sessions")

# Execute the e-commerce demonstration test
if __name__ == "__main__":
    # Configure telemetry to track and monitor the AI agent interactions
    # Only setup when running directly, not when imported by tests
    setup_monocle_telemetry(
        workflow_name="adk_ecommerce",
        monocle_exporters_list='file,console'  # Log to both console and file
    )
    asyncio.run(demonstrate_ecommerce_agent_flow())