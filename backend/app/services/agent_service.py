"""
Agent Service: Orchestrates the AI Shopping Agent with controlled backend tools.

Responsibilities:
- Manages agent session state (structured, not just conversation history)
- Executes tool calls: search_products, get_product, get_cart, add_to_cart
- Enforces merchant/customer isolation via trusted session context
- Uses Gemini's function calling for tool orchestration
- Never exposes internal implementation details to the LLM
"""

import os
import json
import logging
import uuid
from decimal import Decimal
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from google import genai
from google.genai import types
from huggingface_hub import InferenceClient
from sqlalchemy.orm import Session

from app.core.config import settings

logger = logging.getLogger(__name__)
from app.core.merchant_context import MerchantContext
from app.models.agent_session import AgentSession
from app.models.cart import Cart, CartItem
from app.models.enums import CartStatus
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

CHECKOUT_SINGLE_PRODUCT_TOOL = types.FunctionDeclaration(
    name="checkout_single_product",
    description=(
        "Initiate checkout for a SPECIFIC product from the last search results. "
        "Use when user says 'buy the first one', 'proceed payment of 1st product', "
        "'buy product number 2', 'purchase the third one', 'pay for the second shirt', etc. "
        "These refer to the CURRENT SESSION'S LAST SEARCH RESULTS by position. "
        "NEVER use this for 'checkout my cart' or 'checkout everything' — use checkout_cart instead."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "reference_position": types.Schema(
                type=types.Type.INTEGER,
                minimum=1,
                maximum=50,
                description="1-based position in last search results. Use for 'first', 'second', '1st', '2nd', etc."
            ),
            "product_id": types.Schema(
                type=types.Type.STRING,
                description="Explicit product UUID (alternative to reference_position, only use if user provided exact UUID)"
            ),
            "quantity": types.Schema(
                type=types.Type.INTEGER,
                minimum=1,
                maximum=99,
                description="Quantity to purchase. Extract from 'buy 2 of the first' → quantity=2. Default is 1."
            ),
            "size": types.Schema(
                type=types.Type.STRING,
                description="Size if user specified: 'L', 'M', 'XL', etc. Leave null if user did not specify."
            ),
            "address_hint": types.Schema(
                type=types.Type.STRING,
                description="Address reference if user specified: 'default', 'home', 'office', '2'. Null if not mentioned."
            ),
        },
        required=[],
    ),
)

CHECKOUT_CART_TOOL = types.FunctionDeclaration(
    name="checkout_cart",
    description=(
        "Initiate checkout for the customer's ENTIRE ACTIVE CART. "
        "Use when user says 'checkout my cart', 'proceed with my cart', 'pay for my cart', "
        "'checkout everything', 'buy everything in my cart', or plain 'checkout' with no product reference."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "address_hint": types.Schema(
                type=types.Type.STRING,
                description="Address reference if user specified: 'default', 'home', 'office', '2'. Null if not mentioned."
            ),
        },
        required=[],
    ),
)

CONFIRM_CHECKOUT_TOOL = types.FunctionDeclaration(
    name="confirm_checkout",
    description=(
        "Called ONLY when the user explicitly confirms a pending checkout. "
        "The checkout summary has already been shown. The user has said 'yes', 'proceed', 'confirm', etc. "
        "This creates the order record."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "confirm": types.Schema(
                type=types.Type.BOOLEAN,
                description="Must be true. User confirmed they want to proceed."
            ),
        },
        required=["confirm"],
    ),
)

REMOVE_FROM_CART_TOOL = types.FunctionDeclaration(
    name="remove_from_cart",
    description=(
        "Remove an item or all items from the customer's active cart. "
        "Use when user says 'remove the first item from my cart', 'delete item #2', "
        "'remove the shirt from cart', 'empty my cart', or 'clear my cart'."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "item_position": types.Schema(
                type=types.Type.INTEGER,
                minimum=1,
                maximum=50,
                description="1-based position of the item in the customer's cart (e.g. 1 for 1st cart item, 2 for 2nd cart item)."
            ),
            "product_name": types.Schema(
                type=types.Type.STRING,
                description="Product name or keyword to match and remove from cart (e.g. 'shirt', 'pants', 'Oxford')."
            ),
            "cart_item_id": types.Schema(
                type=types.Type.STRING,
                description="Specific cart item UUID if known."
            ),
            "remove_all": types.Schema(
                type=types.Type.BOOLEAN,
                description="Set to true if user wants to clear or empty the entire cart."
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
    REMOVE_FROM_CART_TOOL,
    CHECKOUT_SINGLE_PRODUCT_TOOL,
    CHECKOUT_CART_TOOL,
    CONFIRM_CHECKOUT_TOOL,
])

# ---------------------------------------------------------------------------
# Tool Function Declarations for Hugging Face / OpenAI Schema
# ---------------------------------------------------------------------------

HF_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search for products using natural language query. Returns a list of products with position indices for follow-up references.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query (e.g., 'black formal shirt under 2500')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 10, max 50). Use when user specifies a count like 'top 5' or 'show me 3'.",
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product",
            "description": "Get detailed information about a specific product by ID. Use when user asks for details about a product from search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Product UUID from search results",
                    },
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cart",
            "description": "Get the current customer's cart contents with items, quantities, prices, and totals.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_cart",
            "description": "Add a product variant to the customer's cart using a position reference from the most recent search results. Use this when user says 'add the first one', 'add second product', etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reference_position": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "description": "Position in the most recent search results (1-based). Use this for 'add the first one', 'add second product', etc.",
                    },
                    "product_id": {
                        "type": "string",
                        "description": "Product UUID (alternative to reference_position, use when you have the ID from get_product)",
                    },
                    "variant_id": {
                        "type": "string",
                        "description": "Product variant UUID (required if using product_id)",
                    },
                    "quantity": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 99,
                        "default": 1,
                        "description": "Quantity to add",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_from_cart",
            "description": "Remove an item or all items from the customer's cart. Use when user says 'remove item #1', 'remove the shirt from my cart', 'empty my cart', etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_position": {
                        "type": "integer",
                        "description": "1-based position of the item in the cart (e.g. 1 for first item, 2 for second item).",
                    },
                    "product_name": {
                        "type": "string",
                        "description": "Name or keyword of the product to remove from cart (e.g. 'shirt', 'pants').",
                    },
                    "cart_item_id": {
                        "type": "string",
                        "description": "Specific cart item UUID if known.",
                    },
                    "remove_all": {
                        "type": "boolean",
                        "description": "True to remove all items and empty the cart.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "checkout_single_product",
            "description": (
                "Initiate checkout for a SPECIFIC product from the last search results. "
                "Use when user says 'buy the first one', 'proceed payment of 1st product', "
                "'buy product number 2', 'purchase the third one', etc. "
                "These refer to the CURRENT SESSION'S LAST SEARCH RESULTS by position. "
                "NEVER use this for 'checkout my cart' or 'checkout everything'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reference_position": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "description": "1-based position in last search results. Use for 'first', 'second', '1st', '2nd', etc.",
                    },
                    "product_id": {
                        "type": "string",
                        "description": "Explicit product UUID (only if user provided exact UUID)",
                    },
                    "quantity": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 99,
                        "default": 1,
                        "description": "Quantity. Extract from 'buy 2 of the first' → quantity=2.",
                    },
                    "size": {
                        "type": "string",
                        "description": "Size if user specified: 'L', 'M', 'XL'. Null if not specified.",
                    },
                    "address_hint": {
                        "type": "string",
                        "description": "Address reference: 'default', 'home', 'office', '2'. Null if not mentioned.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "checkout_cart",
            "description": (
                "Initiate checkout for the customer's ENTIRE ACTIVE CART. "
                "Use when user says 'checkout my cart', 'proceed with my cart', 'pay for my cart', "
                "'checkout everything', 'buy everything in my cart', or plain 'checkout' with no product reference."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "address_hint": {
                        "type": "string",
                        "description": "Address reference: 'default', 'home', 'office', '2'. Null if not mentioned.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_checkout",
            "description": (
                "Called ONLY when the user explicitly confirms a pending checkout. "
                "The checkout summary has already been shown. User said 'yes', 'proceed', 'confirm', etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "confirm": {
                        "type": "boolean",
                        "description": "Must be true. User confirmed they want to proceed.",
                    },
                },
                "required": ["confirm"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# System Instruction for the Agent
# ---------------------------------------------------------------------------

AGENT_SYSTEM_INSTRUCTION = """You are a helpful AI Shopping Assistant for a merchant's store.

Your capabilities:
1. Search products using natural language queries
2. Get detailed product information
3. View the customer's cart
4. Add products to the cart
5. Checkout a specific search result product (checkout_single_product)
6. Checkout the entire cart (checkout_cart)

Core Rules:
- Be concise and helpful
- Use the tools provided — do NOT invent product information, prices, IDs, or stock
- When user refers to "the second one" or "first product", use the position from the most recent search results
- Do not ask for product IDs — use search result positions
- If a tool fails, explain the actual reason to the user
- Do not mention internal implementation, embeddings, database queries, or system internals
===== SEARCH / BROWSE =====

When the user is looking for, browsing, or asking about products (e.g., "show me shirts", "find black sneakers", "white formal shirts", "what products do you have?"):
1. Call search_products with the user's search query.
2. NEVER call add_to_cart, checkout_single_product, or checkout_cart during a search/browse query.
3. ONLY call add_to_cart if the user explicitly asks to add an item to their cart.

===== ADD TO CART =====

For "add the second one to cart", "add first product", "add #1", "put the first shirt in my cart":
1. Call add_to_cart with reference_position=<that position number> and quantity (default 1)
2. The system resolves the actual product_id and variant_id from that position
3. If user specifies a variant like "add the second one in Large":
   - Call get_product first with the product_id from that position
   - Find the Large variant, call add_to_cart with specific variant_id
4. If user specifies quantity "add 2 of the second one", use quantity=2. Default is 1.
5. NEVER invent product_id values like "1" or "first" — positions are integers, IDs are UUIDs.

===== REMOVE FROM CART =====

When the user asks to remove an item or empty their cart:
- "remove the first item", "remove item #1 from cart", "delete #2 from cart" → call remove_from_cart(item_position=<N>)
- "remove the shirt from my cart", "remove black shoes from cart" → call remove_from_cart(product_name="<name>")
- "empty my cart", "clear my cart", "remove everything from cart" → call remove_from_cart(remove_all=true)
- After removing an item, inform the user which item was removed and current cart count.

===== PURCHASE / CHECKOUT — THREE DISTINCT MODES =====

MODE 1 — SINGLE PRODUCT CHECKOUT (checkout_single_product):

Use when user says:
  "buy the first one"
  "proceed payment of 1st product"
  "checkout the first product"
  "purchase the second shirt"
  "buy product number 3"
  "pay for the first one"
  "I want the 2nd one"

These ALWAYS refer to the LAST SEARCH RESULTS by position number.
Call: checkout_single_product(reference_position=<N>, quantity=<qty>, size=<size if mentioned>, address_hint=<hint if mentioned>)

CRITICAL: Do NOT call get_cart or checkout_cart for single-product purchases.
The backend will resolve position → actual product UUID. Do NOT invent UUIDs.

Examples:
  "proceed payment of 1st product"
    → checkout_single_product(reference_position=1)

  "buy 2 of the first one"
    → checkout_single_product(reference_position=1, quantity=2)

  "buy first shirt in Large"
    → checkout_single_product(reference_position=1, size="L")

  "buy first product and use my default address"
    → checkout_single_product(reference_position=1, address_hint="default")

  "buy first product, use my office address"
    → checkout_single_product(reference_position=1, address_hint="office")

MODE 2 — CART CHECKOUT (checkout_cart):

Use when user says:
  "checkout my cart"
  "proceed with my cart"
  "pay for my cart"
  "checkout everything"
  "buy everything in my cart"

Call: checkout_cart(address_hint=<hint if mentioned>)

MODE 3 — GENERIC CHECKOUT → defaults to CART:

If user just says "checkout" with NO product reference:
  → checkout_cart()

===== CHECKOUT STATE MACHINE =====

After calling checkout_single_product or checkout_cart, the backend may respond with:
1. "needs_variant_selection: true" — the product has multiple sizes/variants and user hasn't chosen.
   → Ask the user: "What size would you like? Available: S, M, L, XL"
   → When user responds with size, call checkout_single_product again with size=<their answer>

2. "needs_address: true" — customer has no saved address.
   → Ask the user: "What delivery address would you like to use?"
   → Collect: name, address line 1, city, state, postal code, country
   → The backend will save and use it.

3. "awaiting_confirmation: true" — checkout summary shown.
   → Ask: "Shall I proceed to payment?" or present the summary and ask for confirmation.
   → When user says "yes" / "proceed" / "confirm" → call confirm_checkout(confirm=true)
   → When user says "no" / "cancel" → tell them checkout was cancelled.

DO NOT create an order without explicit user confirmation.
DO NOT say "payment successful" — only the backend can verify payments.

===== ADDRESS RULES =====

- "use my default address" → address_hint="default"
- "use my home address" → address_hint="home"
- "use office" / "use my work address" → address_hint="office"
- "use address 2" → address_hint="2"
- No address mentioned → address_hint=null (backend resolves default if exists)

DO NOT ask for address if user already specified one or if they have a default address that the backend has found.

===== ORDINAL EXTRACTION =====

Extract positions from natural language:
  "first" / "1st" / "#1" / "number 1" → reference_position=1
  "second" / "2nd" / "#2" / "number 2" → reference_position=2
  "third" / "3rd" / "#3" / "number 3" → reference_position=3
  "last" → position of last search result

===== QUANTITY =====

Extract from:
  "buy 2 of the first one" → quantity=2
  "buy the first one" → quantity=1 (default)
  "checkout 3 of the second product" → quantity=3
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
        provider: Optional[str] = None,
    ) -> None:
        self.db = db
        self.merchant_context = merchant_context
        self.customer = customer
        self._intent_service = intent_service
        self._product_service = product_service

        if provider:
            self.provider = provider.lower()
        elif os.getenv("LLM_PROVIDER"):
            self.provider = os.getenv("LLM_PROVIDER").lower()
        else:
            self.provider = (
                getattr(settings, "LLM_PROVIDER", None)
                or "huggingface"
            ).lower()

        hf_token = (
            getattr(settings, "HF_TOKEN", None)
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
            or os.getenv("HUGGINGFACE_API_KEY")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        gemini_key = (
            getattr(settings, "GEMINI_API_KEY", None)
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )

        if not provider:
            explicit_env_provider = os.getenv("LLM_PROVIDER")
            # If user explicitly set LLM_PROVIDER in .env, honor it! Only fallback if provider is entirely unspecified
            if not explicit_env_provider and not getattr(settings, "LLM_PROVIDER", None):
                if self.provider == "huggingface" and not hf_token and gemini_key:
                    self.provider = "gemini"
                elif self.provider == "gemini" and not gemini_key and hf_token:
                    self.provider = "huggingface"

        # Initialize clients
        self._gemini_client = None
        self._hf_client = None

    @property
    def intent_service(self) -> IntentService:
        if self._intent_service is None:
            self._intent_service = IntentService(provider=self.provider)
        return self._intent_service

    @property
    def product_service(self) -> ProductService:
        if self._product_service is None:
            self._product_service = ProductService(self.db)
        return self._product_service

    @property
    def gemini_client(self):
        if self._gemini_client is None:
            api_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not configured")
            self._gemini_client = genai.Client(api_key=api_key)
        return self._gemini_client

    @property
    def hf_client(self):
        if self._hf_client is None:
            token = (
                getattr(settings, "HF_TOKEN", None)
                or os.getenv("HF_TOKEN")
                or os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
                or os.getenv("HUGGINGFACE_API_KEY")
                or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            )
            if not token:
                raise ValueError(
                    "HF_TOKEN is not configured. "
                    "Please set HF_TOKEN or HUGGINGFACEHUB_ACCESS_TOKEN in your environment or .env file."
                )
            self._hf_client = InferenceClient(
                token=token,
                base_url=getattr(settings, "HF_API_BASE", None),
            )
        return self._hf_client

    # ---------------------------------------------------------------------------
    # Session Management
    # ---------------------------------------------------------------------------

    def get_or_create_session(self, session_id: str) -> AgentSession:
        """Get existing session or create a new one, linking active customer cart."""
        session = self.db.query(AgentSession).filter(
            AgentSession.session_id == session_id,
            AgentSession.merchant_id == self.merchant_context.merchant_id
        ).first()

        if session:
            # Update customer_id if not set and we have a customer now
            if session.customer_id is None and self.customer is not None:
                session.customer_id = self.customer.id
                self.db.flush()
            # If session has no cart_id but customer has an active cart, link it!
            if self.customer and not session.cart_id:
                active_cart = self.db.query(Cart).filter(
                    Cart.customer_id == self.customer.id,
                    Cart.status == CartStatus.ACTIVE
                ).first()
                if active_cart:
                    session.cart_id = active_cart.id
                    self.db.flush()
            return session

        # Create new session
        session = AgentSession(
            session_id=session_id,
            merchant_id=self.merchant_context.merchant_id,
            customer_id=self.customer.id if self.customer else None,
        )

        # Link existing active cart if customer already has one
        if self.customer:
            active_cart = self.db.query(Cart).filter(
                Cart.customer_id == self.customer.id,
                Cart.status == CartStatus.ACTIVE
            ).first()
            if active_cart:
                session.cart_id = active_cart.id

        self.db.add(session)
        self.db.flush()
        return session

    def _get_cart(self, session: AgentSession) -> Cart:
        """Get or create the customer's active cart."""
        if session.cart_id:
            cart = self.db.query(Cart).filter(
                Cart.id == session.cart_id,
                Cart.status == CartStatus.ACTIVE,
            ).first()
            if cart:
                return cart

        # Resolve customer context
        effective_customer_id = session.customer_id or (self.customer.id if self.customer else None)

        if not effective_customer_id and getattr(settings, "DEBUG", False):
            fallback_customer = self.db.query(Customer).filter(
                Customer.merchant_id == self.merchant_context.merchant_id
            ).order_by(Customer.created_at.asc()).first()
            if fallback_customer:
                effective_customer_id = fallback_customer.id
                session.customer_id = fallback_customer.id

        if not effective_customer_id:
            raise ValueError("Customer not logged in. Please provide an X-Customer-Id header to access cart.")

        cart = self.db.query(Cart).filter(
            Cart.customer_id == effective_customer_id,
            Cart.status == CartStatus.ACTIVE
        ).first()

        if not cart:
            cart = Cart(customer_id=effective_customer_id)
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

        # Cart summary context
        cart_to_summarize = None
        if session.cart_id:
            cart_to_summarize = self.db.query(Cart).filter(Cart.id == session.cart_id).first()
        elif self.customer:
            cart_to_summarize = self.db.query(Cart).filter(
                Cart.customer_id == self.customer.id,
                Cart.status == CartStatus.ACTIVE
            ).first()
            if cart_to_summarize:
                session.cart_id = cart_to_summarize.id

        if cart_to_summarize and cart_to_summarize.items:
            cart_descs = []
            for idx, item in enumerate(cart_to_summarize.items, 1):
                variant = item.variant
                product_title = variant.product.title if variant and variant.product else "Item"
                color_info = f" {variant.color}" if variant and variant.color else ""
                size_info = f" {variant.size}" if variant and variant.size else ""
                cart_descs.append(f"  [Cart Item #{idx}] {item.quantity}x {product_title}{color_info}{size_info} (price: ₹{item.price_at_addition}, item_id: {item.id})")
            parts.append(f"Customer Cart ({len(cart_to_summarize.items)} items):\n" + "\n".join(cart_descs))
        elif cart_to_summarize:
            parts.append("Customer Cart: empty (0 items)")

        if self.customer:
            parts.append(f"Customer: {self.customer.name} ({self.customer.email})")

        # Checkout state context: tell the LLM if there's a pending checkout
        if session.checkout_state:
            cs = session.checkout_state
            step = cs.get("step", "unknown")
            mode = cs.get("mode", "unknown")
            parts.append(
                f"PENDING CHECKOUT: mode={mode}, step={step}. "
                f"If the user confirms (yes/proceed/confirm), call confirm_checkout(confirm=true). "
                f"If they cancel, tell them checkout was cancelled and clear state."
            )

        return "\n".join(parts) if parts else "New session."

    def _generate_with_tools(
        self,
        user_message: str,
        context: str,
        session: AgentSession,
    ) -> Dict[str, Any]:
        """Generate response using configured LLM provider with function calling and automatic fallback."""
        provider = getattr(self, "provider", None)
        if not provider:
            if getattr(self, "_gemini_client", None) is not None:
                provider = "gemini"
            elif getattr(self, "_hf_client", None) is not None:
                provider = "huggingface"
            else:
                provider = getattr(settings, "LLM_PROVIDER", "huggingface").lower()

        if provider == "gemini":
            try:
                return self._generate_with_tools_gemini(user_message, context, session)
            except Exception as e:
                logger.warning(f"Gemini agent failed: {e}. Checking for Hugging Face fallback...")
                hf_token = (
                    getattr(settings, "HF_TOKEN", None)
                    or os.getenv("HF_TOKEN")
                    or os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
                    or os.getenv("HUGGINGFACE_API_KEY")
                )
                if hf_token:
                    try:
                        return self._generate_with_tools_huggingface(user_message, context, session)
                    except Exception as hf_err:
                        logger.error(f"Hugging Face fallback also failed: {hf_err}")
                raise
        else:
            try:
                return self._generate_with_tools_huggingface(user_message, context, session)
            except Exception as e:
                logger.warning(f"Hugging Face agent failed: {e}. Checking for Gemini fallback...")
                gemini_key = (
                    getattr(settings, "GEMINI_API_KEY", None)
                    or os.getenv("GEMINI_API_KEY")
                    or os.getenv("GOOGLE_API_KEY")
                )
                if gemini_key:
                    try:
                        return self._generate_with_tools_gemini(user_message, context, session)
                    except Exception as gem_err:
                        logger.error(f"Gemini fallback also failed: {gem_err}")
                raise

    def _generate_with_tools_huggingface(
        self,
        user_message: str,
        context: str,
        session: AgentSession,
    ) -> Dict[str, Any]:
        """Generate response using Hugging Face Inference API with tool calling."""
        prompt = f"""Context:
{context}

User: {user_message}

IMPORTANT ROUTING RULES (use context above for positions/IDs):

ADD TO CART:
- "add the first one", "add #1", "add second one", etc. → add_to_cart(reference_position=N)
- If user specifies variant (e.g. "in Large") → call get_product first, then add_to_cart with variant_id
- Explicit UUID provided → call get_product, then add_to_cart with variant_id
- NEVER use product_id="1" for "first product" — position is an integer, IDs are UUIDs

PURCHASE / CHECKOUT:
- "buy first product", "proceed payment of 1st product", "purchase the first one", "pay for 2nd item", "checkout the third one"
  → call checkout_single_product(reference_position=N)
  * N = ordinal from user: "1st"/"first"=1, "2nd"/"second"=2, "third"=3, etc.
  * Do NOT call get_cart or checkout_cart for single-item purchases
  * The backend resolves position → actual product UUID — do NOT invent UUIDs
- "checkout my cart", "checkout everything", "pay for everything in cart" → call checkout_cart()
- Plain "checkout" with no product reference → call checkout_cart()
- User confirms pending checkout ("yes", "proceed", "confirm") → call confirm_checkout(confirm=true)
- Do NOT call search_products unless user asks for a NEW search
"""
        primary_model = getattr(settings, "HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
        candidate_models = [primary_model, "Qwen/Qwen2.5-72B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"]
        candidate_models = [m for m in dict.fromkeys(candidate_models) if m]

        messages = [
            {"role": "system", "content": AGENT_SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ]

        last_exc = None
        for model_name in candidate_models:
            for attempt in range(2):
                try:
                    response = self.hf_client.chat_completion(
                        model=model_name,
                        messages=messages,
                        tools=HF_AGENT_TOOLS,
                        tool_choice="auto",
                        temperature=0.0,
                        max_tokens=600,
                    )
                    return self._process_response_huggingface(response, session)
                except Exception as exc:
                    last_exc = exc
                    err_str = str(exc)
                    if "429" in err_str or "Rate limit" in err_str or "temporarily unavailable" in err_str:
                        if model_name != candidate_models[-1]:
                            break
                        continue
                    raise

        if last_exc:
            raise last_exc
        return {"text": "I'm having trouble processing that. Please try again.", "tool_calls": [], "tool_results": []}

    def _process_response_huggingface(self, response, session: AgentSession) -> Dict[str, Any]:
        """Process Hugging Face response and execute any tool calls."""
        result = {
            "text": "",
            "tool_calls": [],
            "tool_results": [],
        }

        if not response or not response.choices:
            result["text"] = "I'm having trouble processing that. Please try again."
            return result

        choice = response.choices[0]
        message = choice.message

        if message.content:
            result["text"] = message.content.strip()

        if message.tool_calls:
            for tc in message.tool_calls:
                func_name = getattr(tc.function, "name", None) or (tc.function.get("name") if isinstance(tc.function, dict) else None)
                raw_args = getattr(tc.function, "arguments", None) or (tc.function.get("arguments") if isinstance(tc.function, dict) else {})
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args) if raw_args.strip() else {}
                    except Exception:
                        args = {}
                elif isinstance(raw_args, dict):
                    args = raw_args
                else:
                    args = {}

                if func_name:
                    result["tool_calls"].append({
                        "name": func_name,
                        "args": args,
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

    def _generate_with_tools_gemini(
        self,
        user_message: str,
        context: str,
        session: AgentSession,
    ) -> Dict[str, Any]:
        """Generate response using Gemini with function calling."""
        prompt = f"""Context:
{context}

User: {user_message}

IMPORTANT ROUTING RULES (use context above for positions/IDs):

SEARCH / BROWSE:
- "show me shirts", "white shirts", "find black pants", "what do you have", "search shoes" → call ONLY search_products(query=...)
- DO NOT call add_to_cart or checkout tools when user is searching or browsing products.

ADD TO CART:
- "add the first one", "add #1", "add second one", "put #2 in my cart", etc. → add_to_cart(reference_position=N)
- If user specifies variant (e.g. "in Large") → call get_product first, then add_to_cart with variant_id
- Explicit UUID provided → call get_product, then add_to_cart with variant_id
- NEVER use product_id="1" for "first product" — position is an integer, IDs are UUIDs

REMOVE FROM CART:
- "remove the first item", "remove item #1 from my cart", "delete #2" → call remove_from_cart(item_position=N)
- "remove shirt from cart", "remove sneakers from my cart" → call remove_from_cart(product_name="...")
- "empty cart", "clear my cart", "remove everything from cart" → call remove_from_cart(remove_all=true)

PURCHASE / CHECKOUT:
- "buy first product", "proceed payment of 1st product", "purchase the first one", "pay for 2nd item", "checkout the third one"
  → call checkout_single_product(reference_position=N)
  * N = ordinal from user: "1st"/"first"=1, "2nd"/"second"=2, "third"=3, etc.
  * Do NOT call get_cart or checkout_cart for single-item purchases
  * The backend resolves position → actual product UUID — do NOT invent UUIDs
- "checkout my cart", "checkout everything", "pay for everything in cart" → call checkout_cart()
- Plain "checkout" with no product reference → call checkout_cart()
- User confirms pending checkout ("yes", "proceed", "confirm") → call confirm_checkout(confirm=true)
- Do NOT call search_products unless user asks for a NEW search
"""

        config = types.GenerateContentConfig(
            system_instruction=AGENT_SYSTEM_INSTRUCTION,
            tools=[AGENT_TOOLS],
            temperature=0.0,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        primary_model = getattr(settings, "GEMINI_INTENT_MODEL", None) or os.getenv("GEMINI_INTENT_MODEL", "gemini-3.6-flash")
        candidate_models = [primary_model, "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash"]
        candidate_models = [m for m in dict.fromkeys(candidate_models) if m]

        last_exc = None
        for model_name in candidate_models:
            for attempt in range(2):
                try:
                    response = self.gemini_client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=config,
                    )
                    return self._process_response(response, session)
                except Exception as exc:
                    last_exc = exc
                    err_str = str(exc)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        if model_name != candidate_models[-1]:
                            break
                        continue
                    raise

        if last_exc:
            raise last_exc
        return {"text": "I'm having trouble processing that. Please try again.", "tool_calls": [], "tool_results": []}

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

        elif tool_name == "remove_from_cart":
            return self._execute_remove_from_cart(args, session)

        # New purchase-scope tools
        elif tool_name == "checkout_single_product":
            return self._execute_checkout_single_product(args, session)

        elif tool_name == "checkout_cart":
            return self._execute_checkout_cart(args, session)

        elif tool_name == "confirm_checkout":
            return self._execute_confirm_checkout(args, session)

        # Legacy: keep old "checkout" name for backwards compat in case LLM still uses it
        elif tool_name == "checkout":
            return self._execute_checkout_cart(args, session)

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

    def _execute_remove_from_cart(self, args: Dict[str, Any], session: AgentSession) -> Dict[str, Any]:
        """Execute remove_from_cart with support for position, name matching, ID, or clear all."""
        cart = self._get_cart(session)
        if not cart or not cart.items:
            return {
                "success": False,
                "error": "Cart is empty",
                "message": "Your cart is currently empty.",
            }

        remove_all = args.get("remove_all", False)
        item_position = args.get("item_position")
        item_id_str = args.get("cart_item_id") or args.get("item_id")
        product_query = args.get("product_name") or args.get("query")

        # 1. Clear entire cart
        if remove_all or (product_query and str(product_query).lower() in ("all", "everything", "cart", "all items")):
            count = len(cart.items)
            for item in list(cart.items):
                self.db.delete(item)
            self.db.commit()
            return {
                "success": True,
                "removed_count": count,
                "message": f"Removed all {count} item(s) from your cart. Your cart is now empty.",
            }

        target_item = None

        # 2. Match by direct cart item UUID
        if item_id_str:
            try:
                target_uuid = uuid.UUID(str(item_id_str).strip())
                target_item = next((i for i in cart.items if i.id == target_uuid), None)
            except Exception:
                pass

        # 3. Match by 1-based position in cart
        if not target_item and item_position is not None:
            try:
                pos = int(item_position) - 1
                if 0 <= pos < len(cart.items):
                    target_item = cart.items[pos]
            except Exception:
                pass

        # 4. Match by product title / color keyword
        if not target_item and product_query:
            pq = str(product_query).lower().strip()
            for item in cart.items:
                title = (item.variant.product.title if item.variant and item.variant.product else "").lower()
                color = (item.variant.color or "").lower() if item.variant else ""
                if pq in title or pq in color or any(w in title for w in pq.split()):
                    target_item = item
                    break

        if not target_item:
            return {
                "success": False,
                "error": "Item not found",
                "message": "Could not find that item in your cart to remove.",
            }

        removed_title = target_item.variant.product.title if target_item.variant and target_item.variant.product else "Item"
        self.db.delete(target_item)
        self.db.commit()

        cart_info = self.tool_get_cart(session)

        return {
            "success": True,
            "removed_item": removed_title,
            "cart_summary": cart_info,
            "message": f"Removed '{removed_title}' from your cart. You now have {cart_info.get('item_count', 0)} item(s) in your cart.",
        }

    def _execute_checkout_single_product(
        self, args: Dict[str, Any], session: AgentSession
    ) -> Dict[str, Any]:
        """
        Execute checkout for a SINGLE product from last search results.

        Flow:
        1. Resolve product from reference_position or explicit product_id
        2. Check variant selection needed
        3. Resolve address
        4. If address needed, ask user
        5. Show summary → store checkout_state as awaiting_confirmation
        6. Return summary to user for confirmation
        """
        from app.services.purchase_intent_resolver import (
            PurchaseIntentResolver, PurchaseIntentError, InsufficientInventoryError
        )
        from app.services.address_service import AddressService
        from app.services.checkout_service import SingleProductCheckoutService

        reference_position = args.get("reference_position")
        product_id_str = args.get("product_id")
        quantity = int(args.get("quantity", 1))
        size = args.get("size")
        address_hint = args.get("address_hint")

        logger.info(
            f"[PURCHASE_DEBUG] ACTION=CHECKOUT PURCHASE_SCOPE=SINGLE_PRODUCT "
            f"REFERENCE_POSITION={reference_position} PRODUCT_ID_STR={product_id_str} "
            f"QUANTITY={quantity} SIZE={size} ADDRESS_HINT={address_hint}"
        )

        # ---- Step 1: Resolve product + variant ----
        try:
            resolver = PurchaseIntentResolver(
                db=self.db,
                merchant_id=self.merchant_context.merchant_id,
            )
            target = resolver.resolve_single_product(
                session=session,
                reference_position=reference_position,
                product_id_str=product_id_str,
                quantity=quantity,
                size=size,
            )
        except PurchaseIntentError as e:
            logger.warning(f"[PURCHASE_DEBUG] RESOLVE_ERROR={e}")
            return {"success": False, "error": str(e), "message": str(e)}
        except Exception as e:
            logger.error(f"[PURCHASE_DEBUG] UNEXPECTED_RESOLVE_ERROR={e}", exc_info=True)
            return {"success": False, "error": str(e), "message": f"Could not resolve product: {e}"}

        # ---- Step 2: Variant selection needed? ----
        if target.needs_variant_selection:
            sizes = [v["size"] for v in target.available_variants if v.get("size")]
            colors = [v["color"] for v in target.available_variants if v.get("color")]
            options = sizes or colors
            return {
                "success": True,
                "needs_variant_selection": True,
                "product_title": target.product.title,
                "available_variants": target.available_variants,
                "message": (
                    f"What size would you like for '{target.product.title}'? "
                    f"Available: {', '.join(options)}"
                    if options else
                    f"Please select a variant for '{target.product.title}'."
                ),
            }

        logger.info(
            f"[PURCHASE_DEBUG] RESOLVED_PRODUCT_ID={target.product.id} "
            f"RESOLVED_VARIANT_ID={target.variant.id} "
            f"UNIT_PRICE={target.unit_price} QUANTITY={quantity}"
        )

        # ---- Step 3: Resolve address ----
        address = None
        address_ref = "none"
        if self.customer:
            addr_service = AddressService(self.db)
            address = addr_service.resolve_address_hint(self.customer.id, address_hint)
            address_ref = str(address.id) if address else "none"

        logger.info(f"[PURCHASE_DEBUG] ADDRESS_SELECTION={address_ref}")

        # ---- Step 4: Need address from user? ----
        if not address and self.customer:
            # Store partial checkout state so next message can continue
            session.checkout_state = {
                "mode": "SINGLE_PRODUCT",
                "step": "awaiting_address",
                "resolved_product_id": str(target.product.id),
                "resolved_variant_id": str(target.variant.id),
                "quantity": quantity,
                "address_id": None,
            }
            self.db.flush()
            return {
                "success": True,
                "needs_address": True,
                "message": "What delivery address would you like to use? Please provide: name, address line 1, city, state, postal code, and country.",
            }

        # ---- Step 5: Build and store checkout summary ----
        checkout_svc = SingleProductCheckoutService(
            db=self.db,
            merchant_context=self.merchant_context,
            customer=self.customer,
        )
        summary = checkout_svc.build_summary(
            product=target.product,
            variant=target.variant,
            quantity=quantity,
            address=address,
        )
        total = target.unit_price * quantity

        logger.info(f"[PURCHASE_DEBUG] CHECKOUT_TOTAL={total} ADDRESS_ID={address_ref}")

        # Store state for confirmation step
        session.checkout_state = {
            "mode": "SINGLE_PRODUCT",
            "step": "awaiting_confirmation",
            "resolved_product_id": str(target.product.id),
            "resolved_variant_id": str(target.variant.id),
            "quantity": quantity,
            "address_id": str(address.id) if address else None,
            "summary": summary,
        }
        self.db.flush()

        return {
            "success": True,
            "awaiting_confirmation": True,
            "checkout_summary": summary,
            "message": (
                f"Here's your order summary:\n"
                f"  Product: {target.product.title}"
                + (f" ({target.variant.color or ''} {target.variant.size or ''}).strip()" if (target.variant.color or target.variant.size) else "")
                + f"\n  Quantity: {quantity}\n"
                f"  Unit price: ₹{target.unit_price}\n"
                f"  Total: ₹{total}\n"
                + (f"  Delivery to: {summary.get('delivery_address')}\n" if summary.get('delivery_address') else "  No delivery address set.\n")
                + "\nShall I proceed to payment?"
            ),
        }

    def _execute_checkout_cart(
        self, args: Dict[str, Any], session: AgentSession
    ) -> Dict[str, Any]:
        """
        Execute checkout for the ENTIRE active cart.

        Flow:
        1. Load and validate cart
        2. Resolve address
        3. Show summary → store checkout_state as awaiting_confirmation
        """
        from app.services.checkout_service import CheckoutService, CartValidationResult
        from app.services.address_service import AddressService

        address_hint = args.get("address_hint")

        logger.info(
            f"[PURCHASE_DEBUG] ACTION=CHECKOUT PURCHASE_SCOPE=CART ADDRESS_HINT={address_hint}"
        )

        try:
            checkout_service = CheckoutService(
                db=self.db,
                merchant_context=self.merchant_context,
                customer=self.customer,
            )
            cart = checkout_service.get_or_create_active_cart(session.session_id)
            validation = checkout_service.validate_and_calculate(cart)
        except Exception as e:
            return {"success": False, "error": str(e), "message": str(e)}

        # Resolve address
        address = None
        address_ref = "none"
        if self.customer:
            addr_service = AddressService(self.db)
            address = addr_service.resolve_address_hint(self.customer.id, address_hint)
            address_ref = str(address.id) if address else "none"

        logger.info(
            f"[PURCHASE_DEBUG] CART_ITEM_COUNT={len(validation.items)} "
            f"CART_SUBTOTAL={validation.subtotal} ADDRESS_SELECTION={address_ref}"
        )

        if not address and self.customer:
            session.checkout_state = {
                "mode": "CART",
                "step": "awaiting_address",
                "address_id": None,
            }
            self.db.flush()
            return {
                "success": True,
                "needs_address": True,
                "message": "What delivery address would you like to use? Please provide: name, address line 1, city, state, postal code, and country.",
            }

        # Build address display
        address_display = None
        if address:
            parts = [
                address.recipient_name or (self.customer.name if self.customer else ""),
                address.address_line_1,
                f"{address.city}, {address.state} {address.postal_code}",
                address.country,
            ]
            address_display = ", ".join(p for p in parts if p)

        logger.info(f"[PURCHASE_DEBUG] CHECKOUT_TOTAL={validation.subtotal}")

        summary = {
            "item_count": len(validation.items),
            "subtotal": str(validation.subtotal),
            "total": str(validation.subtotal),
            "currency": "INR",
            "delivery_address": address_display,
            "address_id": str(address.id) if address else None,
            "warnings": validation.warnings,
        }

        session.checkout_state = {
            "mode": "CART",
            "step": "awaiting_confirmation",
            "address_id": str(address.id) if address else None,
            "summary": summary,
        }
        self.db.flush()

        return {
            "success": True,
            "awaiting_confirmation": True,
            "checkout_summary": summary,
            "message": (
                f"Here's your cart summary:\n"
                f"  {len(validation.items)} item(s), total: ₹{validation.subtotal}\n"
                + (f"  Delivery to: {address_display}\n" if address_display else "  No delivery address set.\n")
                + (f"  ⚠️ {'; '.join(validation.warnings)}\n" if validation.warnings else "")
                + "\nShall I proceed to payment?"
            ),
        }

    def _execute_confirm_checkout(
        self, args: Dict[str, Any], session: AgentSession
    ) -> Dict[str, Any]:
        """
        Execute the final confirmed checkout step.
        Creates the Order (and address snapshot).
        Razorpay NOT integrated yet.
        """
        from app.services.checkout_service import (
            CheckoutService, SingleProductCheckoutService, CartValidationResult
        )
        from app.services.address_service import AddressService
        from app.models.product import Product, ProductVariant
        from app.models.customer_address import CustomerAddress

        confirm = args.get("confirm", False)
        if not confirm:
            return {
                "success": False,
                "error": "Confirmation required",
                "message": "Please confirm by saying 'yes' or 'proceed' to complete your order.",
            }

        state = session.checkout_state
        if not state or state.get("step") != "awaiting_confirmation":
            return {
                "success": False,
                "error": "No pending checkout",
                "message": "There is no pending checkout to confirm. Please start checkout again.",
            }

        mode = state.get("mode")
        address_id_str = state.get("address_id")

        # Resolve address object
        address = None
        if address_id_str and self.customer:
            addr_service = AddressService(self.db)
            try:
                address = addr_service.get_address_by_id(self.customer.id, uuid.UUID(address_id_str))
            except Exception:
                pass

        try:
            if mode == "SINGLE_PRODUCT":
                product_id = uuid.UUID(state["resolved_product_id"])
                variant_id = uuid.UUID(state["resolved_variant_id"])
                quantity = int(state.get("quantity", 1))

                product = self.db.query(Product).filter(Product.id == product_id).first()
                variant = self.db.query(ProductVariant).filter(ProductVariant.id == variant_id).first()

                if not product or not variant:
                    return {"success": False, "error": "Product not found", "message": "Could not find the product for this checkout."}

                svc = SingleProductCheckoutService(
                    db=self.db,
                    merchant_context=self.merchant_context,
                    customer=self.customer,
                )
                result = svc.execute_checkout(
                    product=product,
                    variant=variant,
                    quantity=quantity,
                    address=address,
                )

                logger.info(
                    f"[PURCHASE_DEBUG] ORDER_CREATED MODE=SINGLE_PRODUCT ORDER_ID={result.order_id} "
                    f"TOTAL={result.total_amount}"
                )

                # Clear checkout state
                session.checkout_state = None
                self.db.flush()

                return {
                    "success": True,
                    "order_id": str(result.order_id),
                    "total_amount": str(result.total_amount),
                    "currency": result.currency,
                    "status": result.status,
                    "message": (
                        f"Order placed successfully! Order ID: {result.order_id}. "
                        f"Total: ₹{result.total_amount}. "
                        f"Payment integration coming soon — your order is confirmed and pending payment."
                    ),
                }

            elif mode == "CART":
                checkout_service = CheckoutService(
                    db=self.db,
                    merchant_context=self.merchant_context,
                    customer=self.customer,
                )
                cart = checkout_service.get_or_create_active_cart(session.session_id)
                validation = checkout_service.validate_and_calculate(cart)
                order = checkout_service.create_order(
                    cart=cart,
                    validation_result=validation,
                    address=address,
                    customer_name=self.customer.name if self.customer else None,
                )
                self.db.commit()

                logger.info(
                    f"[PURCHASE_DEBUG] ORDER_CREATED MODE=CART ORDER_ID={order.id} "
                    f"TOTAL={order.total_amount}"
                )

                # Clear checkout state
                session.checkout_state = None
                self.db.flush()

                return {
                    "success": True,
                    "order_id": str(order.id),
                    "total_amount": str(order.total_amount),
                    "currency": "INR",
                    "status": "pending_payment",
                    "message": (
                        f"Order placed successfully! Order ID: {order.id}. "
                        f"Total: ₹{order.total_amount}. "
                        f"Payment integration coming soon — your order is confirmed and pending payment."
                    ),
                }

            else:
                return {
                    "success": False,
                    "error": "Invalid checkout state",
                    "message": "Unexpected checkout mode. Please start checkout again.",
                }

        except Exception as e:
            self.db.rollback()
            logger.error(f"[PURCHASE_DEBUG] CONFIRM_CHECKOUT_ERROR={e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": f"Could not complete the order: {e}",
            }

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

            elif name in ("add_to_cart", "remove_from_cart"):
                if res.get("success"):
                    summary_parts.append(res.get("message", "Cart updated."))
                else:
                    summary_parts.append(f"Could not update cart: {res.get('message', 'Unknown error')}")

            elif name in ("checkout_single_product", "checkout_cart", "confirm_checkout"):
                if res.get("message"):
                    summary_parts.append(res.get("message"))
                elif res.get("error"):
                    summary_parts.append(f"Checkout error: {res.get('error')}")

            elif name == "checkout":
                if res.get("success"):
                    amount_inr = res.get("amount_paise", 0) / 100
                    summary_parts.append(
                        f"Checkout initiated successfully. Your total is ₹{amount_inr:.2f}. "
                        f"Razorpay order created: {res.get('razorpay_order_id')}. "
                        f"Please complete payment on the frontend using the provided order details."
                    )
                else:
                    summary_parts.append(f"Checkout failed: {res.get('message', 'Unknown error')}")

        prompt = f"""Based on these tool results, provide a concise, helpful response to the customer:

{chr(10).join(summary_parts)}

Rules:
- Be concise and natural
- Don't mention tool names or internal details
- If there are multiple results, mention them briefly
- If there's an error, explain it clearly
"""

        fallback_text = "\n".join(summary_parts) if summary_parts else "Done."

        provider = getattr(self, "provider", None)
        if not provider:
            if getattr(self, "_gemini_client", None) is not None:
                provider = "gemini"
            elif getattr(self, "_hf_client", None) is not None:
                provider = "huggingface"
            else:
                provider = getattr(settings, "LLM_PROVIDER", "huggingface").lower()

        if provider == "gemini":
            try:
                config = types.GenerateContentConfig(
                    system_instruction=AGENT_SYSTEM_INSTRUCTION,
                    temperature=0.0,
                )
                primary_model = getattr(settings, "GEMINI_INTENT_MODEL", None) or "gemini-3.6-flash"
                candidate_models = [primary_model, "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash"]
                candidate_models = [m for m in dict.fromkeys(candidate_models) if m]

                for model_name in candidate_models:
                    try:
                        followup_response = self.gemini_client.models.generate_content(
                            model=model_name,
                            contents=prompt,
                            config=config,
                        )
                        if followup_response and followup_response.text:
                            return followup_response.text
                    except Exception:
                        continue
            except Exception:
                pass

            # Try Hugging Face fallback if configured
            hf_token = getattr(settings, "HF_TOKEN", None) or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
            if hf_token:
                try:
                    model_name = getattr(settings, "HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
                    messages = [
                        {"role": "system", "content": AGENT_SYSTEM_INSTRUCTION},
                        {"role": "user", "content": prompt},
                    ]
                    followup_response = self.hf_client.chat_completion(
                        model=model_name,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=400,
                    )
                    if followup_response and followup_response.choices and followup_response.choices[0].message:
                        return followup_response.choices[0].message.content or fallback_text
                except Exception:
                    pass

            return fallback_text
        else:
            try:
                primary_model = getattr(settings, "HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
                candidate_models = [primary_model, "Qwen/Qwen2.5-72B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"]
                candidate_models = [m for m in dict.fromkeys(candidate_models) if m]
                messages = [
                    {"role": "system", "content": AGENT_SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ]

                for model_name in candidate_models:
                    try:
                        followup_response = self.hf_client.chat_completion(
                            model=model_name,
                            messages=messages,
                            temperature=0.0,
                            max_tokens=400,
                        )
                        if followup_response and followup_response.choices and followup_response.choices[0].message:
                            return followup_response.choices[0].message.content or fallback_text
                    except Exception:
                        continue
            except Exception:
                pass

            # Try Gemini fallback if configured
            gemini_key = getattr(settings, "GEMINI_API_KEY", None) or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if gemini_key:
                try:
                    config = types.GenerateContentConfig(
                        system_instruction=AGENT_SYSTEM_INSTRUCTION,
                        temperature=0.0,
                    )
                    followup_response = self.gemini_client.models.generate_content(
                        model=getattr(settings, "GEMINI_INTENT_MODEL", "gemini-3.6-flash"),
                        contents=prompt,
                        config=config,
                    )
                    if followup_response and followup_response.text:
                        return followup_response.text
                except Exception:
                    pass

            return fallback_text

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
        checkout_summary = None
        checkout_state_out = None
        needs_variant_selection = False
        available_variants: List[Dict[str, Any]] = []

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

            elif name in ("add_to_cart", "remove_from_cart") and res.get("success"):
                cart_updated = True

            elif name in ("checkout_single_product", "checkout_cart") and res.get("success"):
                if res.get("needs_variant_selection"):
                    needs_variant_selection = True
                    available_variants = res.get("available_variants", [])
                elif res.get("awaiting_confirmation"):
                    checkout_summary = res.get("checkout_summary")
                    checkout_state_out = session.checkout_state

            elif name == "confirm_checkout" and res.get("success"):
                checkout_summary = {
                    "order_id": res.get("order_id"),
                    "total_amount": res.get("total_amount"),
                    "currency": res.get("currency"),
                    "status": res.get("status"),
                }

            # Legacy: old "checkout" tool (Razorpay flow - keep for backwards compat)
            elif name == "checkout" and res.get("success"):
                checkout_summary = {
                    "order_id": res.get("order_id"),
                    "razorpay_order_id": res.get("razorpay_order_id"),
                    "amount_paise": res.get("amount_paise"),
                    "currency": res.get("currency"),
                    "key_id": res.get("key_id"),
                    "status": res.get("status"),
                }

        if cart_updated and cart_summary is None:
            cart_res = self.tool_get_cart(session)
            if cart_res.get("success"):
                cart_summary = {
                    "cart_id": cart_res.get("cart_id"),
                    "items": cart_res.get("items", []),
                    "subtotal": cart_res.get("subtotal"),
                    "item_count": cart_res.get("item_count"),
                    "warnings": cart_res.get("warnings", []),
                }

        return AgentChatResponse(
            session_id=session.session_id,
            message=result.get("text", ""),
            products=products,
            cart_updated=cart_updated,
            product_detail=product_detail,
            cart_summary=cart_summary,
            checkout_summary=checkout_summary,
            checkout_state=checkout_state_out,
            needs_variant_selection=needs_variant_selection,
            available_variants=available_variants,
        )