import type { CartItemData } from '../api/client';

interface CartPanelProps {
  items: CartItemData[];
  onClose: () => void;
  onCheckout: () => void;
  onRemoveItem?: (cartItemId: string) => void;
}

export function CartPanel({ items, onClose, onCheckout, onRemoveItem }: CartPanelProps) {
  const subtotal = items.reduce((sum, item) => sum + (Number(item.price) * item.quantity), 0);

  return (
    <div className="cart-panel">
      <div className="cart-header">
        <h3>Shopping Cart ({items.reduce((s, i) => s + i.quantity, 0)})</h3>
        <button className="close-button" onClick={onClose}>×</button>
      </div>

      <div className="cart-content">
        {items.length === 0 ? (
          <div className="cart-empty">
            <p>Your cart is empty</p>
            <p>Ask the AI assistant to browse products and add them to your cart!</p>
          </div>
        ) : (
          <>
            <div className="cart-items">
              {items.map((item) => (
                <div key={item.cart_item_id || item.id} className="cart-item">
                  <div className="cart-item-info">
                    <h4>{item.title}</h4>
                    <div className="cart-item-meta">
                      {item.color && <span>Color: {item.color}</span>}
                      {item.size && <span>Size: {item.size}</span>}
                      <span>Qty: {item.quantity}</span>
                    </div>
                    <div className="cart-item-price">
                      ₹{(Number(item.price) * item.quantity).toFixed(2)}
                    </div>
                  </div>
                  {onRemoveItem && item.cart_item_id && (
                    <button
                      className="cart-item-remove"
                      onClick={() => onRemoveItem(item.cart_item_id)}
                      title="Remove item"
                    >
                      🗑️
                    </button>
                  )}
                </div>
              ))}
            </div>

            <div className="cart-summary">
              <div className="cart-subtotal">
                <span>Subtotal</span>
                <span>₹{subtotal.toFixed(2)}</span>
              </div>
            </div>
          </>
        )}
      </div>

      {items.length > 0 && (
        <div className="cart-footer">
          <button className="btn-checkout" onClick={onCheckout}>
            Proceed to Checkout
          </button>
        </div>
      )}
    </div>
  );
}