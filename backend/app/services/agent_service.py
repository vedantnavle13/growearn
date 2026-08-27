"""
Agent Service: Orchestrates the AI Shopping Agent with controlled backend tools.

Responsibilities:
- Manages agent session state (structured, not just conversation history)
- Executes tool calls: search_products, get_product, get_cart, add_to_cart
- Enforces merchant/customer isolation via trusted session context
- Uses Gemini's function calling for tool orchestration
- Never exposes internal implementation details to the LLM
"""

import json
import uuid
from decimal import Decimal
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from google import genai
from google.genai import types
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.merchant_context import MerchantContext
from app.models.agent_session import AgentSession
from app.models.cart import Cart, CartItem
from app.models.customer import Customer
from app.models.product import Product, ProductVariant
from app.schemas.agent import (
    AgentChatResponse,
    AgentProductSummary,
    AgentToolSearchInput,
    AgentToolGetProductInput,
    AgentToolAddToCartInput,
)
from app.schemas.intent import CommerceIntent
from app.schemas.product import IntentSearchResponse, IntentSearchResult
from app.services.intent_service import IntentService
from app.services.product_service import ProductService

if TYPE_CHECKING:
    from app.models.merchant import Merchant


# ---------------------------------------------------------------------------
# Tool Function Declarations for Gemini
# ---------------------------------------------------------------------------

SEARCH_PRODUCTS_TOOL = types.FunctionDeclaration(
    name="search_products",
    description="Search for products using natural language query. Returns a list of products with position indices for follow-up references.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "query": types.Schema(
                type=types.Type.STRING,
                description="Natural language search query (e.g., 'black formal shirt under 2500')"
            ),
            "limit": types.Schema(
                type=types.Type.INTEGER,
                description="Maximum number of results to return (default 10, max 50). Use when user specifies a count like 'top 5' or 'show me 3'.",
                minimum=1,
                maximum=50,
            ),
        },
        required=["query"],
    ),
)

GET_PRODUCT_TOOL = types.FunctionDeclaration(
    name="get_product",
    description="Get detailed information about a specific product by ID. Use when user asks for details about a product from search results.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "product_id": types.Schema(
                type=types.Type.STRING,
                format="uuid",
                description="Product UUID from search results"
            ),
        },
        required=["product_id"],
    ),
)

GET_CART_TOOL = types.FunctionDeclaration(
    name="get_cart",
    description="Get the current customer's cart contents with items, quantities, prices, and totals.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={},
    ),
)

ADD_TO_CART_TOOL = types.FunctionDeclaration(
    name="add_to_cart",
    description="Add a product variant to the customer's cart using a position reference from the most recent search results. Use this when user says 'add the first one', 'add second product', etc.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "reference_position": types.Schema(
                type=types.Type.INTEGER,
                minimum=1,
                maximum=50,
                description="Position in the most recent search results (1-based). Use this for 'add the first one', 'add second product', etc."
            ),
            "product_id": types.Schema(
                type=types.Type.STRING,
                format="uuid",
                description="Product UUID (alternative to reference_position, use when you have the ID from get_product)"
            ),
            "variant_id": types.Schema(
                type=types.Type.STRING,
                format="uuid",
                description="Product variant UUID (required if using product_id)"
            ),
            "quantity": types.Schema(
                type=types.Type.INTEGER,
                minimum=1,
                maximum=99,
                default=1,
                description="Quantity to add"
            ),
        },
        required=[],
    ),
)

AGENT_TOOLS = types.Tool(function_declarations=[
    SEARCH_PRODUCTS_TOOL,
    GET_PRODUCT_TOOL,
    GET_CART_TOOL,
    ADD_TO_CART_TOOL,
])


# ---------------------------------------------------------------------------
# System Instruction for the Agent
# ---------------------------------------------------------------------------

AGENT_SYSTEM_INSTRUCTION = """You are a helpful AI Shopping Assistant for a merchant's store.

Your capabilities:
1. Search products using natural language queries
2. Get detailed product information
3. View the customer's cart
4. Add products to the cart

Rules:
- Be concise and helpful
- Use the tools provided - do NOT invent product information, prices, or stock
- When user refers to "the second one" or "first product", use the position from the most recent search results
- Do not ask for product IDs - use the search results stored in your context
- If a tool fails, explain the actual reason to the user
- Do not mention internal implementation, embeddings, database queries, or system internals
- If user asks unrelated questions, respond naturally without calling product tools

When showing search results, the system will provide position numbers (1, 2, 3...). Use these for follow-ups.

CRITICAL: Adding to cart uses the reference_position parameter:

For "add the second one to cart", "add first product", "add #1", "put the first shirt in my cart":
1. Look at the LAST SEARCH RESULTS in the context above. Find the position number (e.g., position 1 for "first", 2 for "second").
2. Call add_to_cart with reference_position=<that position number> and quantity (default 1).
3. The system will automatically resolve the actual product_id and variant_id from that position.
4. If the user specifies a variant like "add the second one in Large", call get_product first with the product_id from that position, then call add_to_cart with the specific variant_id.
5. If user specifies quantity like "add 2 of the second one", use quantity=2. Default is 1.

Example flow for "add the second one to cart":
- Context shows: "2. Black Shirt - 1999 black M (in stock) (variant_id: abc-123)"
- Call add_to_cart(reference_position=2, quantity=1)

Example flow for "add 3 of the first one":
- Context shows: "1. Blue Jeans - 2999 blue 32 (in stock) (variant_id: xyz-789)"
- Call add_to_cart(reference_position=1, quantity=3)

Example flow for "add the second one in Large":
- Context shows position 2 has product_id="xyz-789"
- Call get_product(product_id="xyz-789") to see all variants
- Find the Large variant, note its variant_id
- Call add_to_cart(product_id="xyz-789", variant_id="<large_variant_id>", quantity=1)

The search results context includes variant_id for the primary variant shown. Use reference_position for unambiguous add-to-cart from search results.
"""


class AgentService:
    """
    Orchestrates the AI Shopping Agent with session management and tool execution.
    """

    def __init__(
        self,
        db: Session,
        merchant_context: MerchantContext,
        customer: Optional[Customer] = None,
        intent_service: Optional[IntentService] = None,
        product_service: Optional[ProductService] = None,
    ) -> None:
        self.db = db
        self.merchant_context = merchant_context
        self.customer = customer
        self._intent_service = intent_service
        self._product_service = product_service

        # Initialize Gemini client
        self._gemini_client = None

    @property
    def intent_service(self) -> IntentService:
        if self._intent_service is None:
            self._intent_service = IntentService()
        return self._intent_service

    @property
    def product_service(self) -> ProductService:
        if self._product_service is None:
            self._product_service = ProductService(self.db)
        return self._product_service

    @property
    def gemini_client(self):
        if self._gemini_client is None:
            api_key = settings.GEMINI_API_KEY
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not configured")
            self._gemini_client = genai.Client(api_key=api_key)
        return self._gemini_client

    # ---------------------------------------------------------------------------
    # Session Management
    # ---------------------------------------------------------------------------

    def get_or_create_session(self, session_id: str) -> AgentSession:
        """Get existing session or create a new one."""
        session = self.db.query(AgentSession).filter(
            AgentSession.session_id == session_id,
            AgentSession.merchant_id == self.merchant_context.merchant_id
        ).first()

        if session:
            return session

        # Create new session
        session = AgentSession(
            session_id=session_id,
            merchant_id=self.merchant_context.merchant_id,
            customer_id=self.customer.id if self.customer else None,
        )
        self.db.add(session)
        self.db.flush()
        return session

    def _get_cart(self, session: AgentSession) -> Cart:
        """Get or create the customer's cart."""
        if session.cart_id:
            cart = self.db.query(Cart).filter(
                Cart.id == session.cart_id,
                Cart.customer_id == session.customer_id
            ).first()
            if cart:
                return cart

        # Create or get active cart for customer
        if not session.customer_id:
            # Guest cart - we'll handle this with session-based cart
            # For MVP, require customer_id (logged in)
            raise ValueError("Customer not logged in. Cannot access cart.")

        cart = self.db.query(Cart).filter(
            Cart.customer_id == session.customer_id,
            Cart.status == "active"
        ).first()

        if not cart:
            cart = Cart(customer_id=session.customer_id)
            self.db.add(cart)
            self.db.flush()

        session.cart_id = cart.id
        self.db.flush()
        return cart

    # ---------------------------------------------------------------------------
    # Tool Implementations
    # ---------------------------------------------------------------------------

    def tool_search_products(self, query: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """Tool: search_products - Uses intent-driven search pipeline."""
        try:
            # Use the full intent search pipeline
            # Apply limit from tool call, or default to config value
            search_limit = limit if limit is not None else settings.DEFAULT_SEARCH_LIMIT
            # Clamp to maximum allowed
            search_limit = min(search_limit, settings.MAX_SEARCH_LIMIT)

            response: IntentSearchResponse = self.product_service.intent_search_products(
                merchant_id=self.merchant_context.merchant_id,
                raw_query=query,
                customer_id=self.customer.id if self.customer else None,
                limit=search_limit,
            )

            # Build compact results with position for follow-up reference
            products = []
            for idx, result in enumerate(response.results):
                # Get primary variant for display
                primary_variant = result.variants[0] if result.variants else None
                products.append(AgentProductSummary(
                    id=result.id,
                    title=result.title,
                    price=result.price,
                    color=primary_variant.color if primary_variant else None,
                    size=primary_variant.size if primary_variant else None,
                    in_stock=primary_variant.in_stock if primary_variant else False,
                    position=idx + 1,
                    variant_id=primary_variant.id if primary_variant else None,
                ))

            # Return structured result for LLM and session storage
            return {
                "success": True,
                "query": query,
                "intent": response.intent,
                "total": response.total,
                "products": [p.model_dump(mode="json") for p in products],
                "message": f"Found {response.total} product(s) for '{query}'",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Search failed. Please try again.",
            }

    def tool_get_product(self, product_id: uuid.UUID) -> Dict[str, Any]:
        """Tool: get_product - Returns detailed product info after merchant validation."""
        try:
            # Fetch product with variants and inventory, scoped to merchant
            product = self.db.query(Product).filter(
                Product.id == product_id,
                Product.merchant_id == self.merchant_context.merchant_id,
                Product.is_active.is_(True),
            ).first()

            if not product:
                return {
                    "success": False,
                    "error": "Product not found",
                    "message": "Product not found or not available in this store.",
                }

            # Build detailed response
            variants = []
            for variant in product.variants:
                available = 0
                if variant.inventory:
                    available = variant.inventory.quantity - variant.inventory.reserved_quantity
                variants.append({
                    "id": str(variant.id),
                    "sku": variant.sku,
                    "size": variant.size,
                    "color": variant.color,
                    "price": str(variant.price),
                    "in_stock": available > 0,
                    "available_quantity": available,
                })

            return {
                "success": True,
                "product": {
                    "id": str(product.id),
                    "title": product.title,
                    "description": product.description,
                    "price": str(product.price),
                    "attributes": product.attributes,
                    "variants": variants,
                },
                "message": f"Details for '{product.title}'",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to get product details.",
            }

    def tool_get_cart(self, session: AgentSession) -> Dict[str, Any]:
        """Tool: get_cart - Returns cart contents with totals."""
        try:
            cart = self._get_cart(session)

            # Load cart items with variants and product info
            cart_items = self.db.query(CartItem).filter(
                CartItem.cart_id == cart.id
            ).all()

            items = []
            subtotal = Decimal("0")
            warnings = []

            for item in cart_items:
                # Get current variant state
                variant = self.db.query(ProductVariant).filter(
                    ProductVariant.id == item.variant_id
                ).first()

                if not variant:
                    warnings.append(f"Item {item.id}: variant no longer exists")
                    continue

                product = self.db.query(Product).filter(
                    Product.id == variant.product_id,
                    Product.merchant_id == self.merchant_context.merchant_id
                ).first()

                if not product:
                    warnings.append(f"Item {item.id}: product not found")
                    continue

                # Check current availability
                available = 0
                if variant.inventory:
                    available = variant.inventory.quantity - variant.inventory.reserved_quantity

                if available < item.quantity:
                    warnings.append(f"'{product.title}' ({variant.color or ''} {variant.size or ''}): only {available} available, you have {item.quantity} in cart")

                line_total = item.price_at_addition * item.quantity
                subtotal += line_total

                items.append({
                    "cart_item_id": str(item.id),
                    "product_id": str(product.id),
                    "variant_id": str(variant.id),
                    "product_title": product.title,
                    "variant_sku": variant.sku,
                    "color": variant.color,
                    "size": variant.size,
                    "quantity": item.quantity,
                    "unit_price": str(item.price_at_addition),
                    "line_total": str(line_total),
                    "in_stock": available >= item.quantity,
                    "available_quantity": available,
                })

            return {
                "success": True,
                "cart_id": str(cart.id),
                "items": items,
                "subtotal": str(subtotal),
                "item_count": len(items),
                "warnings": warnings,
                "message": f"Cart has {len(items)} item(s), subtotal: {subtotal}",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to get cart.",
            }

    def tool_add_to_cart(
        self,
        session: AgentSession,
        product_id: uuid.UUID,
        variant_id: uuid.UUID,
        quantity: int = 1,
    ) -> Dict[str, Any]:
        """Tool: add_to_cart - Adds validated product/variant to cart."""
        try:
            # 1. Validate product belongs to merchant
            product = self.db.query(Product).filter(
                Product.id == product_id,
                Product.merchant_id == self.merchant_context.merchant_id,
                Product.is_active.is_(True),
            ).first()

            if not product:
                return {
                    "success": False,
                    "error": "Product not found",
                    "message": "Product not found in this store.",
                }

            # 2. Validate variant belongs to product
            variant = self.db.query(ProductVariant).filter(
                ProductVariant.id == variant_id,
                ProductVariant.product_id == product_id,
            ).first()

            if not variant:
                return {
                    "success": False,
                    "error": "Invalid variant",
                    "message": "Selected variant does not belong to this product.",
                }

            # 3. Check availability
            available = 0
            if variant.inventory:
                available = variant.inventory.quantity - variant.inventory.reserved_quantity

            if available <= 0:
                return {
                    "success": False,
                    "error": "Out of stock",
                    "message": f"'{product.title}' ({variant.color or ''} {variant.size or ''}) is out of stock.",
                }

            if quantity > available:
                return {
                    "success": False,
                    "error": "Insufficient stock",
                    "message": f"Only {available} unit(s) available for '{product.title}' ({variant.color or ''} {variant.size or ''}). You requested {quantity}.",
                }

            # 4. Get or create cart
            cart = self._get_cart(session)

            # 5. Check if item already in cart
            existing_item = self.db.query(CartItem).filter(
                CartItem.cart_id == cart.id,
                CartItem.variant_id == variant_id,
            ).first()

            if existing_item:
                new_quantity = existing_item.quantity + quantity
                if new_quantity > available:
                    return {
                        "success": False,
                        "error": "Insufficient stock",
                        "message": f"Total quantity would be {new_quantity}, but only {available} available.",
                    }
                existing_item.quantity = new_quantity
                existing_item.price_at_addition = variant.price  # Update to current price
                self.db.flush()
                item = existing_item
            else:
                # Create new cart item with current price from DB
                item = CartItem(
                    cart_id=cart.id,
                    variant_id=variant_id,
                    quantity=quantity,
                    price_at_addition=variant.price,
                )
                self.db.add(item)
                self.db.flush()

            self.db.commit()

            return {
                "success": True,
                "cart_item_id": str(item.id),
                "product_title": product.title,
                "variant_color": variant.color,
                "variant_size": variant.size,
                "quantity": item.quantity,
                "unit_price": str(variant.price),
                "message": f"Added {quantity} x '{product.title}' ({variant.color or ''} {variant.size or ''}) to cart.",
            }

        except Exception as e:
            self.db.rollback()
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to add to cart.",
            }

    # ---------------------------------------------------------------------------
    # Main Orchestration
    # ---------------------------------------------------------------------------

    def chat(self, message: str, session_id: str) -> AgentChatResponse:
        """Main entry point: process user message and return agent response."""
        # Get or create session
        session = self.get_or_create_session(session_id)

        # Prepare conversation history for Gemini
        # We'll pass the session state as context
        context = self._build_context(session)

        # Generate response with function calling
        response = self._generate_with_tools(message, context, session)

        # Update session with new state
        self._update_session_from_tools(session, response)

        self.db.commit()

        # Build final response
        return self._build_response(response, session)

    def _build_context(self, session: AgentSession) -> str:
        """Build context string for the LLM from session state."""
        parts = []

        if session.current_intent:
            parts.append(f"Current search intent: {json.dumps(session.current_intent)}")

        if session.last_search_results:
            parts.append("Last search results:")
            for p in session.last_search_results:
                variant_info = f" (variant_id: {p['variant_id']})" if p.get('variant_id') else ""
                color_info = f" {p['color']}" if p.get('color') else ""
                size_info = f" {p['size']}" if p.get('size') else ""
                parts.append(
                    f"  {p['position']}. {p['title']} - {p['price']}{color_info}{size_info} "
                    f"({'in stock' if p['in_stock'] else 'out of stock'}){variant_info}"
                )

        if session.cart_id:
            parts.append(f"Active cart: {session.cart_id}")

        if self.customer:
            parts.append(f"Customer: {self.customer.name} ({self.customer.email})")

        return "\n".join(parts) if parts else "New session."

    def _generate_with_tools(
        self,
        user_message: str,
        context: str,
        session: AgentSession,
    ) -> Dict[str, Any]:
        """Generate response using Gemini with function calling."""
        # Build the prompt with context
        prompt = f"""Context:
{context}

User: {user_message}

IMPORTANT: The context above shows the LAST SEARCH RESULTS with position numbers (1, 2, 3...) and variant_ids.
- If the user says "add the first one", "add first product", "add #1", "add product 1", "add second one", "add the third product", "put the first shirt in my cart", etc. — use the reference_position parameter.
- Find the position number from the context (1 for first, 2 for second, etc.) and call add_to_cart with reference_position=<that number>.
- If the user specifies a variant (e.g., "add the second one in Large"), call get_product first with the product_id from that position, then call add_to_cart with the specific variant_id.
- Do NOT call search_products again unless the user asks for a NEW search.
- When user says "add X of the [position] one", use quantity=X. Default quantity is 1.
- NEVER try to construct a product_id from the position number (e.g., don't use product_id="1" for "first product").
"""

        config = types.GenerateContentConfig(
            system_instruction=AGENT_SYSTEM_INSTRUCTION,
            tools=[AGENT_TOOLS],
            temperature=0.0,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        # Use the intent model for agent orchestration
        model_name = getattr(settings, "GEMINI_INTENT_MODEL", "gemini-3.6-flash")

        response = self.gemini_client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )

        return self._process_response(response, session)

    def _process_response(self, response, session: AgentSession) -> Dict[str, Any]:
        """Process Gemini response and execute any tool calls."""
        result = {
            "text": "",
            "tool_calls": [],
            "tool_results": [],
        }

        if not response.candidates:
            result["text"] = "I'm having trouble processing that. Please try again."
            return result

        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            result["text"] = "I'm not sure how to respond. Please try again."
            return result

        for part in candidate.content.parts:
            if part.text:
                result["text"] += part.text
            elif part.function_call:
                fc = part.function_call
                result["tool_calls"].append({
                    "name": fc.name,
                    "args": dict(fc.args) if fc.args else {},
                })

        # Execute tool calls
        for tool_call in result["tool_calls"]:
            tool_result = self._execute_tool(tool_call["name"], tool_call["args"], session)
            result["tool_results"].append({
                "name": tool_call["name"],
                "result": tool_result,
            })

        # If tools were called, generate a follow-up response
        if result["tool_results"]:
            result["text"] = self._generate_followup(result, session)

        return result

    def _execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        session: AgentSession,
    ) -> Dict[str, Any]:
        """Execute a tool call and return the result."""
        if tool_name == "search_products":
            query = args.get("query", "")
            limit = args.get("limit")
            return self.tool_search_products(query, limit=limit)

        elif tool_name == "get_product":
            product_id_str = args.get("product_id")
            if not product_id_str:
                return {"success": False, "error": "Missing product_id", "message": "Product ID required."}
            try:
                product_id = uuid.UUID(product_id_str)
            except ValueError:
                return {"success": False, "error": "Invalid product_id", "message": "Invalid product ID format."}
            return self.tool_get_product(product_id)

        elif tool_name == "get_cart":
            return self.tool_get_cart(session)

        elif tool_name == "add_to_cart":
            return self._execute_add_to_cart(args, session)

        else:
            return {"success": False, "error": f"Unknown tool: {tool_name}", "message": "Unknown tool."}

    def _execute_add_to_cart(self, args: Dict[str, Any], session: AgentSession) -> Dict[str, Any]:
        """Execute add_to_cart with reference resolution."""
        reference_position = args.get("reference_position")
        product_id_str = args.get("product_id")
        variant_id_str = args.get("variant_id")
        quantity = args.get("quantity", 1)

        # Resolve product_id and variant_id from reference_position if provided
        if reference_position is not None:
            if not session.last_search_results:
                return {
                    "success": False,
                    "error": "No search results",
                    "message": "No recent search results to reference. Please search for products first.",
                }

            # reference_position is 1-based, convert to 0-based index
            idx = reference_position - 1
            if idx < 0 or idx >= len(session.last_search_results):
                return {
                    "success": False,
                    "error": "Invalid position",
                    "message": f"Position {reference_position} not found in recent search results (only {len(session.last_search_results)} results).",
                }

            search_result = session.last_search_results[idx]
            product_id_str = str(search_result.get("id")) if search_result.get("id") else None
            variant_id_str = str(search_result.get("variant_id")) if search_result.get("variant_id") else None

            if not product_id_str:
                return {
                    "success": False,
                    "error": "Missing product ID",
                    "message": "Search result at position {reference_position} is missing product ID.",
                }

        if not product_id_str or not variant_id_str:
            return {"success": False, "error": "Missing IDs", "message": "Product ID and variant ID required (either via reference_position or directly)."}

        try:
            product_id = uuid.UUID(product_id_str)
            variant_id = uuid.UUID(variant_id_str)
        except ValueError:
            return {"success": False, "error": "Invalid ID format", "message": "Invalid UUID format."}

        return self.tool_add_to_cart(session, product_id, variant_id, quantity)

    def _generate_followup(
        self,
        tool_execution_result: Dict[str, Any],
        session: AgentSession,
    ) -> str:
        """Generate a natural language follow-up after tool execution."""
        # Build a summary of what happened
        summary_parts = []

        for tool_result in tool_execution_result["tool_results"]:
            name = tool_result["name"]
            res = tool_result["result"]

            if name == "search_products":
                if res.get("success"):
                    count = res.get("total", 0)
                    if count == 0:
                        summary_parts.append(f"No products found for '{res.get('query', '')}'.")
                    else:
                        summary_parts.append(f"Found {count} product(s).")
                else:
                    summary_parts.append(f"Search failed: {res.get('message', 'Unknown error')}")

            elif name == "get_product":
                if res.get("success"):
                    product = res.get("product", {})
                    variants = product.get("variants", [])
                    variant_info = []
                    for v in variants:
                        status = "in stock" if v.get("in_stock") else "out of stock"
                        variant_info.append(f"{v.get('color', '')} {v.get('size', '')} ({status})".strip())
                    summary_parts.append(
                        f"Product: {product.get('title')} - {product.get('price')}. "
                        f"Variants: {', '.join(variant_info)}. "
                        f"Description: {product.get('description', 'N/A')}"
                    )
                else:
                    summary_parts.append(f"Could not get product details: {res.get('message', 'Unknown error')}")

            elif name == "get_cart":
                if res.get("success"):
                    items = res.get("items", [])
                    subtotal = res.get("subtotal", "0")
                    warnings = res.get("warnings", [])
                    if not items:
                        summary_parts.append("Your cart is empty.")
                    else:
                        item_descs = []
                        for item in items:
                            item_descs.append(
                                f"{item['quantity']} x {item['product_title']} "
                                f"({item.get('color', '')} {item.get('size', '')}) - {item['line_total']}"
                            )
                        summary_parts.append(
                            f"Cart ({len(items)} items, subtotal: {subtotal}):\n" + "\n".join(item_descs)
                        )
                        if warnings:
                            summary_parts.append("Warnings: " + "; ".join(warnings))
                else:
                    summary_parts.append(f"Could not get cart: {res.get('message', 'Unknown error')}")

            elif name == "add_to_cart":
                if res.get("success"):
                    summary_parts.append(res.get("message", "Added to cart."))
                else:
                    summary_parts.append(f"Could not add to cart: {res.get('message', 'Unknown error')}")

        # Now ask Gemini to formulate a natural response
        prompt = f"""Based on these tool results, provide a concise, helpful response to the customer:

{chr(10).join(summary_parts)}

Rules:
- Be concise and natural
- Don't mention tool names or internal details
- If there are multiple results, mention them briefly
- If there's an error, explain it clearly
"""

        config = types.GenerateContentConfig(
            system_instruction=AGENT_SYSTEM_INSTRUCTION,
            temperature=0.0,
        )

        model_name = getattr(settings, "GEMINI_INTENT_MODEL", "gemini-3.6-flash")

        followup_response = self.gemini_client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )

        return followup_response.text or "Done."

    def _update_session_from_tools(self, session: AgentSession, result: Dict[str, Any]):
        """Update session state based on tool execution results."""
        for tool_result in result.get("tool_results", []):
            name = tool_result["name"]
            res = tool_result["result"]

            if name == "search_products" and res.get("success"):
                # Store search results with positions
                session.current_intent = res.get("intent")
                session.last_search_results = res.get("products", [])

            elif name == "add_to_cart" and res.get("success"):
                # Cart was updated
                pass

    def _build_response(
        self,
        result: Dict[str, Any],
        session: AgentSession,
    ) -> AgentChatResponse:
        """Build the final API response."""
        products = []
        product_detail = None
        cart_summary = None
        cart_updated = False

        for tool_result in result.get("tool_results", []):
            name = tool_result["name"]
            res = tool_result["result"]

            if name == "search_products" and res.get("success"):
                products = [AgentProductSummary(**p) for p in res.get("products", [])]

            elif name == "get_product" and res.get("success"):
                product_detail = res.get("product")

            elif name == "get_cart" and res.get("success"):
                cart_summary = {
                    "cart_id": res.get("cart_id"),
                    "items": res.get("items", []),
                    "subtotal": res.get("subtotal"),
                    "item_count": res.get("item_count"),
                    "warnings": res.get("warnings", []),
                }

            elif name == "add_to_cart" and res.get("success"):
                cart_updated = True

        return AgentChatResponse(
            session_id=session.session_id,
            message=result.get("text", ""),
            products=products,
            cart_updated=cart_updated,
            product_detail=product_detail,
            cart_summary=cart_summary,
        )