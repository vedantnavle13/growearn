import { ProductList } from './ProductList';
import type { ChatMessage as ChatMessageType, AgentProductSummary } from '../types';

interface ChatMessageProps {
  message: ChatMessageType;
  onAddToCart: (product: AgentProductSummary) => void;
  onBuyNow: (product: AgentProductSummary) => void;
}

export function ChatMessage({ message, onAddToCart, onBuyNow }: ChatMessageProps) {
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';

  if (isSystem) {
    return (
      <div className="chat-message system-message">
        <div className="message-content">{message.content}</div>
      </div>
    );
  }

  return (
    <div className={`chat-message ${isUser ? 'user-message' : 'assistant-message'}`}>
      <div className="message-avatar">{isUser ? 'You' : 'AI'}</div>
      <div className="message-content">
        <div className="message-text">{message.content}</div>
        
        {message.products && message.products.length > 0 && (
          <ProductList
            products={message.products}
            onAddToCart={onAddToCart}
            onBuyNow={onBuyNow}
          />
        )}

        {message.checkoutSummary && (
          <div className="checkout-preview">
            <h4>Checkout Summary</h4>
            <pre>{JSON.stringify(message.checkoutSummary, null, 2)}</pre>
          </div>
        )}

        {message.needsVariantSelection && message.availableVariants && message.availableVariants.length > 0 && (
          <div className="variant-selection-prompt">
            <p>Please select a variant:</p>
            <ul>
              {message.availableVariants.map((variant, idx) => (
                <li key={idx}>
                  {variant.size && `Size: ${variant.size}`} {variant.color && `Color: ${variant.color}`}
                  - ₹{variant.price} {variant.inStock ? '(In Stock)' : '(Out of Stock)'}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}