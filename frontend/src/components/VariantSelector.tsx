import type { VariantOption } from '../types';

interface VariantSelectorProps {
  variants: VariantOption[];
  onSelect: (variant: VariantOption) => void;
}

export function VariantSelector({ variants, onSelect }: VariantSelectorProps) {
  // Group variants by size and color
  const sizes = [...new Set(variants.map((v) => v.size).filter(Boolean))] as string[];
  const colors = [...new Set(variants.map((v) => v.color).filter(Boolean))] as string[];

  return (
    <div className="variant-selector">
      <h4>Select Variant</h4>

      {sizes.length > 0 && (
        <div className="variant-group">
          <label>Size:</label>
          <div className="variant-options">
            {sizes.map((size) => {
              const sizeVariants = variants.filter((v) => v.size === size);
              const inStock = sizeVariants.some((v) => v.inStock);
              return (
                <button
                  key={size}
                  className={`variant-option ${inStock ? '' : 'out-of-stock'}`}
                  onClick={() => {
                    const variant = sizeVariants[0];
                    if (variant && inStock) onSelect(variant);
                  }}
                  disabled={!inStock}
                >
                  {size}
                  {!inStock && <span className="stock-badge">Out of Stock</span>}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {colors.length > 0 && (
        <div className="variant-group">
          <label>Color:</label>
          <div className="variant-options">
            {colors.map((color) => {
              const colorVariants = variants.filter((v) => v.color === color);
              const inStock = colorVariants.some((v) => v.inStock);
              return (
                <button
                  key={color}
                  className={`variant-option ${inStock ? '' : 'out-of-stock'}`}
                  onClick={() => {
                    const variant = colorVariants[0];
                    if (variant && inStock) onSelect(variant);
                  }}
                  disabled={!inStock}
                >
                  {color}
                  {!inStock && <span className="stock-badge">Out of Stock</span>}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {sizes.length === 0 && colors.length === 0 && variants.length > 0 && (
        <div className="variant-group">
          <label>Options:</label>
          <div className="variant-options">
            {variants.map((variant) => (
              <button
                key={variant.id}
                className={`variant-option ${variant.inStock ? '' : 'out-of-stock'}`}
                onClick={() => variant.inStock && onSelect(variant)}
                disabled={!variant.inStock}
              >
                {variant.size && `Size: ${variant.size}`} {variant.color && `Color: ${variant.color}`}
                <span className="variant-price">₹{(variant.price / 100).toFixed(2)}</span>
                {!variant.inStock && <span className="stock-badge">Out of Stock</span>}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}