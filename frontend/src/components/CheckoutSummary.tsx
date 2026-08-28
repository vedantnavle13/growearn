export interface CheckoutSummaryProps {
  summary: Record<string, any> | null;
}

export function CheckoutSummary({ summary }: CheckoutSummaryProps) {
  if (!summary) return null;

  const totalAmount = Number(summary.total ?? summary.subtotal ?? 0);
  const items = Array.isArray(summary.items) ? summary.items : [];

  return (
    <div className="checkout-summary">
      <h4>Order Summary</h4>

      <div className="summary-items">
        {/* Single product summary */}
        {summary.product_title && (
          <div className="summary-item">
            <div className="item-info">
              <div className="item-title">{summary.product_title}</div>
              <div className="item-variant">
                {summary.variant_size && <span>Size: {summary.variant_size} </span>}
                {summary.variant_color && <span>Color: {summary.variant_color} </span>}
              </div>
              <div className="item-quantity">Qty: {summary.quantity || 1}</div>
            </div>
            <div className="item-price">
              ₹{Number(summary.subtotal || summary.unit_price || 0).toFixed(2)}
            </div>
          </div>
        )}

        {/* Multi-item cart summary */}
        {items.length > 0 && items.map((item: any, index: number) => {
          const title = item.product?.title || item.title || item.product_title || 'Item';
          const size = item.variant?.size || item.size;
          const color = item.variant?.color || item.color;
          const qty = item.quantity || 1;
          const price = Number(item.lineTotal ?? item.line_total ?? item.unit_price ?? item.price ?? 0);

          return (
            <div key={index} className="summary-item">
              <div className="item-info">
                <div className="item-title">{title}</div>
                <div className="item-variant">
                  {size && <span>Size: {size} </span>}
                  {color && <span>Color: {color} </span>}
                </div>
                <div className="item-quantity">Qty: {qty}</div>
              </div>
              <div className="item-price">
                ₹{price.toFixed(2)}
              </div>
            </div>
          );
        })}

        {/* Generic cart count summary if no items array */}
        {!summary.product_title && items.length === 0 && summary.item_count && (
          <div className="summary-item">
            <div className="item-info">
              <div className="item-title">Cart Checkout ({summary.item_count} items)</div>
            </div>
          </div>
        )}
      </div>

      <div className="summary-totals">
        {summary.subtotal && (
          <div className="total-row">
            <span>Subtotal</span>
            <span>₹{Number(summary.subtotal).toFixed(2)}</span>
          </div>
        )}
        <div className="total-row total">
          <span>Total Amount</span>
          <span>₹{totalAmount.toFixed(2)}</span>
        </div>
      </div>

      {summary.delivery_address && (
        <div className="summary-address">
          <h5>Delivery Address</h5>
          <div className="address-text">
            📍 {summary.delivery_address}
          </div>
        </div>
      )}

      {summary.address && !summary.delivery_address && (
        <div className="summary-address">
          <h5>Deliver to:</h5>
          <div className="address-text">
            {summary.address.label && <strong>{summary.address.label} </strong>}
            {summary.address.recipientName && <div>{summary.address.recipientName}</div>}
            {summary.address.addressLine1 && <div>{summary.address.addressLine1}</div>}
            {summary.address.city && (
              <div>{summary.address.city}, {summary.address.state} {summary.address.postalCode}</div>
            )}
          </div>
        </div>
      )}

      {Array.isArray(summary.warnings) && summary.warnings.length > 0 && (
        <div className="summary-warnings">
          {summary.warnings.map((w: string, i: number) => (
            <p key={i} className="warning-text">⚠️ {w}</p>
          ))}
        </div>
      )}
    </div>
  );
}