import { ProductCard } from './ProductCard';
import type { AgentProductSummary } from '../types';

interface ProductListProps {
  products: AgentProductSummary[];
  onAddToCart: (product: AgentProductSummary) => void;
  onBuyNow: (product: AgentProductSummary) => void;
}

export function ProductList({ products, onAddToCart, onBuyNow }: ProductListProps) {
  if (!products || products.length === 0) return null;

  return (
    <div className="product-list">
      <h4>Found {products.length} product{products.length !== 1 ? 's' : ''}:</h4>
      <div className="product-grid">
        {products.map((product) => (
          <ProductCard
            key={product.id}
            product={product}
            onAddToCart={onAddToCart}
            onBuyNow={onBuyNow}
          />
        ))}
      </div>
    </div>
  );
}