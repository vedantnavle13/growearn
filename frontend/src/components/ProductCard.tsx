import type { AgentProductSummary } from '../types';

interface ProductCardProps {
  product: AgentProductSummary;
  onAddToCart: (product: AgentProductSummary) => void;
  onBuyNow: (product: AgentProductSummary) => void;
}

export function ProductCard({ product, onAddToCart, onBuyNow }: ProductCardProps) {
  const priceInRupees = Number(product.price).toFixed(2);

  return (
    <div className="product-card">
      <div className="product-image">
        {product.image_url ? (
          <img src={product.image_url} alt={product.title} />
        ) : (
          <div className="product-placeholder">
            <span className="placeholder-icon">📦</span>
            <span className="placeholder-text">No Image</span>
          </div>
        )}
      </div>

      <div className="product-info">
        <h4 className="product-title">{product.title}</h4>
        <div className="product-meta">
          <span className="product-price">₹{priceInRupees}</span>
          {product.color && <span className="product-color">Color: {product.color}</span>}
          {product.size && <span className="product-size">Size: {product.size}</span>}
          <span className={`product-stock ${product.in_stock ? 'in-stock' : 'out-of-stock'}`}>
            {product.in_stock ? 'In Stock' : 'Out of Stock'}
          </span>
          <span className="product-position">#{product.position}</span>
        </div>

        <div className="product-actions">
          <button
            className="btn-add-to-cart"
            onClick={() => onAddToCart(product)}
            disabled={!product.in_stock}
          >
            Add to Cart
          </button>
          <button
            className="btn-buy-now"
            onClick={() => onBuyNow(product)}
            disabled={!product.in_stock}
          >
            Buy Now
          </button>
        </div>
      </div>
    </div>
  );
}