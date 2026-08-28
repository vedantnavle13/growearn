import { AgentProductSummary } from '../api/client';

export type MessageRole = 'user' | 'assistant' | 'system';

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: Date;
  products?: AgentProductSummary[];
  checkoutSummary?: Record<string, unknown>;
  checkoutState?: Record<string, unknown>;
  needsVariantSelection?: boolean;
  availableVariants?: Array<Record<string, unknown>>;
}

export interface CartItem {
  product: AgentProductSummary;
  variant: Record<string, unknown> | null;
  quantity: number;
  unitPrice: number;
  lineTotal: number;
}

export interface CartState {
  items: CartItem[];
  subtotal: number;
}

export type PaymentStatus = 'idle' | 'loading' | 'payment_pending' | 'payment_processing' | 'success' | 'failure';

export interface PaymentState {
  status: PaymentStatus;
  error?: string;
  orderId?: string;
  razorpayOrderId?: string;
  amount?: number;
  currency?: string;
  keyId?: string;
}

export interface Address {
  id: string;
  label: string;
  recipientName: string;
  addressLine1: string;
  addressLine2?: string;
  city: string;
  state: string;
  postalCode: string;
  country: string;
  isDefault: boolean;
}

export interface VariantOption {
  id: string;
  size?: string;
  color?: string;
  price: number;
  inStock: boolean;
  stockQuantity?: number;
}

export interface CheckoutSummary {
  items: Array<{
    product: AgentProductSummary;
    variant: VariantOption | null;
    quantity: number;
    unitPrice: number;
    lineTotal: number;
  }>;
  subtotal: number;
  total: number;
  address?: Address;
}