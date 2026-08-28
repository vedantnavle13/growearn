import { useState, useCallback, useEffect, useRef } from 'react';
import { ChatWidget } from './components/ChatWidget';
import { CartPanel } from './components/CartPanel';
import { CheckoutPanel } from './components/CheckoutPanel';
import { PaymentHandler } from './components/PaymentHandler';
import { apiClient, CartItemData } from './api/client';
import { getSessionId, setSessionId, clearSessionId } from './utils/session';
import type { ChatMessage, PaymentState, Address, VariantOption, CheckoutSummary, AgentProductSummary } from './types';

const CHAT_STORAGE_KEY = 'growearn_chat_messages';

function loadInitialMessages(): ChatMessage[] {
  try {
    const stored = localStorage.getItem(CHAT_STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      return parsed.map((m: any) => ({
        ...m,
        timestamp: new Date(m.timestamp),
      }));
    }
  } catch (e) {
    console.error('Failed to load stored chat messages:', e);
  }
  return [];
}

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>(loadInitialMessages);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cartItems, setCartItems] = useState<CartItemData[]>([]);
  const [showCart, setShowCart] = useState(false);
  const [checkoutData, setCheckoutData] = useState<{
    summary: Record<string, any> | null;
    needsVariant: boolean;
    availableVariants: VariantOption[];
    needsAddress: boolean;
    addresses: Address[];
    selectedVariant: VariantOption | null;
    selectedAddress: Address | null;
  } | null>(null);
  const [paymentState, setPaymentState] = useState<PaymentState>({ status: 'idle' });
  const [showPaymentSuccess, setShowPaymentSuccess] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const sessionId = getSessionId();

  // Save messages to localStorage whenever they change
  useEffect(() => {
    try {
      localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(messages));
    } catch (e) {
      console.error('Failed to persist messages:', e);
    }
  }, [messages]);

  // Fetch real cart items from backend
  const refreshCart = useCallback(async () => {
    try {
      const res = await apiClient.getCart();
      if (res && res.items) {
        setCartItems(res.items);
      }
    } catch (e) {
      console.warn('Could not fetch cart:', e);
    }
  }, []);

  // Initial cart load on page load/refresh
  useEffect(() => {
    let active = true;
    apiClient.getCart()
      .then((res) => {
        if (active && res && res.items) {
          setCartItems(res.items);
        }
      })
      .catch((e) => console.warn('Could not fetch cart:', e));
    return () => {
      active = false;
    };
  }, []);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const handleSendMessage = useCallback(async (message: string) => {
    if (!message.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: message,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      const response = await apiClient.agentChat({ message, session_id: sessionId });

      if (response.session_id !== sessionId) {
        setSessionId(response.session_id);
      }

      const assistantMessage: ChatMessage = {
        id: `msg-${Date.now()}-assistant`,
        role: 'assistant',
        content: response.message,
        timestamp: new Date(),
        products: response.products,
        checkoutSummary: response.checkout_summary,
        checkoutState: response.checkout_state,
        needsVariantSelection: response.needs_variant_selection,
        availableVariants: response.available_variants as VariantOption[],
      };

      setMessages((prev) => [...prev, assistantMessage]);

      if (response.cart_updated || response.cart_summary) {
        refreshCart();
      }

      if (response.checkout_summary) {
        const hasAddress = Boolean(
          response.checkout_summary.delivery_address ||
          response.checkout_summary.address_id ||
          response.checkout_summary.address
        );
        setCheckoutData({
          summary: response.checkout_summary,
          needsVariant: Boolean(response.needs_variant_selection),
          availableVariants: (response.available_variants || []) as VariantOption[],
          needsAddress: !hasAddress,
          addresses: [],
          selectedVariant: null,
          selectedAddress: null,
        });
      } else if (response.needs_variant_selection && response.available_variants && response.available_variants.length > 0) {
        setCheckoutData({
          summary: {},
          needsVariant: true,
          availableVariants: response.available_variants as VariantOption[],
          needsAddress: false,
          addresses: [],
          selectedVariant: null,
          selectedAddress: null,
        });
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to send message';
      setError(errorMessage);
      const errorMsg: ChatMessage = {
        id: `msg-${Date.now()}-error`,
        role: 'system',
        content: `Error: ${errorMessage}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  }, [sessionId, isLoading, refreshCart]);

  const handleAddToCart = useCallback(async (product: AgentProductSummary) => {
    try {
      const targetQuery = product.position
        ? `add product #${product.position} to cart`
        : `add "${product.title}" to cart`;
      await handleSendMessage(targetQuery);
      refreshCart();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add to cart');
    }
  }, [handleSendMessage, refreshCart]);

  const handleBuyNow = useCallback(async (product: AgentProductSummary) => {
    try {
      const targetQuery = product.position
        ? `buy product #${product.position}`
        : `buy "${product.title}"`;
      await handleSendMessage(targetQuery);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to initiate checkout');
    }
  }, [handleSendMessage]);

  const handleRemoveCartItem = useCallback(async (cartItemId: string) => {
    try {
      await apiClient.removeCartItem(cartItemId);
      await refreshCart();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to remove item from cart');
    }
  }, [refreshCart]);

  const handleVariantSelect = useCallback(async (variant: VariantOption) => {
    setCheckoutData((prev) => prev ? { ...prev, selectedVariant: variant, needsVariant: false } : null);
    
    try {
      const response = await apiClient.agentChat({
        message: `select ${variant.size || variant.color || 'variant'}`,
        session_id: sessionId,
      });
      if (response.checkout_summary) {
        setCheckoutData((prev) => prev ? {
          ...prev,
          summary: response.checkout_summary as CheckoutSummary,
          needsAddress: true,
        } : null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to select variant');
    }
  }, [sessionId]);

  const handleAddressSelect = useCallback(async (address: Address) => {
    setCheckoutData((prev) => prev ? { ...prev, selectedAddress: address, needsAddress: false } : null);
    
    try {
      const response = await apiClient.agentChat({
        message: `use ${address.label}`,
        session_id: sessionId,
      });
      if (response.checkout_summary) {
        setCheckoutData((prev) => prev ? {
          ...prev,
          summary: response.checkout_summary as CheckoutSummary,
        } : null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to select address');
    }
  }, [sessionId]);

  const handleProceedToPayment = useCallback(async () => {
    if (!checkoutData?.summary) return;

    setPaymentState((prev) => ({ ...prev, status: 'loading' }));

    try {
      const response = await apiClient.checkout({ session_id: sessionId });
      setPaymentState({
        status: 'payment_pending',
        orderId: response.order_id,
        razorpayOrderId: response.razorpay_order_id,
        amount: response.amount,
        currency: response.currency,
        keyId: response.key_id,
      });
    } catch (err) {
      setPaymentState({
        status: 'failure',
        error: err instanceof Error ? err.message : 'Checkout failed',
      });
    }
  }, [checkoutData, sessionId]);

  const handlePaymentComplete = useCallback(async (paymentData: {
    razorpay_payment_id: string;
    razorpay_order_id: string;
    razorpay_signature: string;
  }) => {
    setPaymentState((prev) => ({ ...prev, status: 'payment_processing' }));

    try {
      await apiClient.verifyPayment(paymentData);
      setPaymentState((prev) => ({ ...prev, status: 'success' }));
      setShowPaymentSuccess(true);
      setCheckoutData(null);
      refreshCart();
    } catch (err) {
      setPaymentState({
        status: 'failure',
        error: err instanceof Error ? err.message : 'Payment verification failed',
      });
    }
  }, [refreshCart]);

  const handlePaymentClose = useCallback(() => {
    setPaymentState({ status: 'idle' });
    setShowPaymentSuccess(false);
  }, []);

  const handleAddAddress = useCallback(async (address: Omit<Address, 'id' | 'isDefault'>) => {
    const newAddress: Address = {
      ...address,
      id: `addr-${Date.now()}`,
      isDefault: false,
    };
    setCheckoutData((prev) => prev ? {
      ...prev,
      addresses: [...prev.addresses, newAddress],
      selectedAddress: newAddress,
      needsAddress: false,
    } : null);
  }, []);

  const handleCloseCheckout = useCallback(() => {
    setCheckoutData(null);
  }, []);

  const handleNewChat = useCallback(() => {
    clearSessionId();
    localStorage.removeItem(CHAT_STORAGE_KEY);
    setMessages([]);
    setCheckoutData(null);
    refreshCart();
  }, [refreshCart]);

  const totalCartCount = cartItems.reduce((sum, item) => sum + item.quantity, 0);

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-brand">
          <h1>Growearn AI Shopping</h1>
        </div>
        <div className="header-actions">
          {messages.length > 0 && (
            <button onClick={handleNewChat} className="btn-secondary-header" title="Start a fresh chat session">
              New Chat
            </button>
          )}
          <button onClick={() => setShowCart(!showCart)} className="cart-toggle">
            🛒 Cart ({totalCartCount})
          </button>
        </div>
      </header>

      <main className="app-main">
        <ChatWidget
          messages={messages}
          isLoading={isLoading}
          error={error}
          onSendMessage={handleSendMessage}
          onAddToCart={handleAddToCart}
          onBuyNow={handleBuyNow}
          messagesEndRef={messagesEndRef}
        />

        {showCart && (
          <CartPanel
            items={cartItems}
            onClose={() => setShowCart(false)}
            onRemoveItem={handleRemoveCartItem}
            onCheckout={() => {
              setShowCart(false);
              handleSendMessage("checkout my cart");
            }}
          />
        )}

        {checkoutData && (
          <CheckoutPanel
            data={checkoutData}
            onVariantSelect={handleVariantSelect}
            onAddressSelect={handleAddressSelect}
            onAddAddress={handleAddAddress}
            onProceedToPayment={handleProceedToPayment}
            onClose={handleCloseCheckout}
          />
        )}

        {paymentState.status !== 'idle' && (
          <PaymentHandler
            state={paymentState}
            onComplete={handlePaymentComplete}
            onClose={handlePaymentClose}
          />
        )}

        {showPaymentSuccess && (
          <div className="payment-success-overlay">
            <div className="payment-success-modal">
              <h2>Payment Successful ✓</h2>
              <p>Order #{paymentState.orderId?.slice(0, 8)}</p>
              <p>Total: ₹{((paymentState.amount || 0) / 100).toFixed(2)}</p>
              <button onClick={handlePaymentClose}>View Order</button>
              <button onClick={handlePaymentClose} className="secondary">Continue Shopping</button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;