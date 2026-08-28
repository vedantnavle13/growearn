import { VariantSelector } from './VariantSelector';
import { AddressSelector } from './AddressSelector';
import { CheckoutSummary } from './CheckoutSummary';
import type { VariantOption, Address } from '../types';

interface CheckoutPanelProps {
  data: {
    summary: Record<string, any> | null;
    needsVariant: boolean;
    availableVariants: VariantOption[];
    needsAddress: boolean;
    addresses: Address[];
    selectedVariant: VariantOption | null;
    selectedAddress: Address | null;
  };
  onVariantSelect: (variant: VariantOption) => void;
  onAddressSelect: (address: Address) => void;
  onAddAddress: (address: Omit<Address, 'id' | 'isDefault'>) => void;
  onProceedToPayment: () => void;
  onClose: () => void;
}

export function CheckoutPanel({
  data,
  onVariantSelect,
  onAddressSelect,
  onAddAddress,
  onProceedToPayment,
  onClose,
}: CheckoutPanelProps) {
  if (!data.summary) return null;

  const totalAmount = Number(data.summary.total ?? data.summary.subtotal ?? 0);

  return (
    <div className="checkout-panel">
      <div className="checkout-header">
        <h3>Checkout</h3>
        <button className="close-button" onClick={onClose}>×</button>
      </div>

      <div className="checkout-content">
        {data.needsVariant && data.availableVariants.length > 0 && (
          <VariantSelector
            variants={data.availableVariants}
            onSelect={onVariantSelect}
          />
        )}

        {data.needsAddress && (
          <AddressSelector
            addresses={data.addresses}
            onSelect={onAddressSelect}
            onAddAddress={onAddAddress}
          />
        )}

        <CheckoutSummary summary={data.summary} />

        {!data.needsVariant && !data.needsAddress && (
          <button className="btn-proceed-payment" onClick={onProceedToPayment}>
            Pay ₹{totalAmount.toFixed(2)} with Razorpay
          </button>
        )}
      </div>
    </div>
  );
}