import axios, { AxiosInstance, AxiosError } from 'axios';
import { config } from '../config';

export interface AgentChatRequest {
  message: string;
  session_id: string;
}

export interface AgentProductSummary {
  id: string;
  title: string;
  price: number;
  color?: string;
  size?: string;
  in_stock: boolean;
  position: number;
  variant_id?: string;
  image_url?: string;
}

export interface AgentChatResponse {
  session_id: string;
  message: string;
  products: AgentProductSummary[];
  cart_updated: boolean;
  product_detail?: Record<string, unknown>;
  cart_summary?: Record<string, unknown>;
  checkout_summary?: Record<string, unknown>;
  checkout_state?: Record<string, unknown>;
  needs_variant_selection: boolean;
  available_variants: Array<Record<string, unknown>>;
}

export interface CheckoutRequest {
  session_id: string;
}

export interface CheckoutResponse {
  order_id: string;
  razorpay_order_id: string;
  amount: number;
  currency: string;
  key_id: string;
  status: string;
}

export interface PaymentVerificationRequest {
  razorpay_order_id: string;
  razorpay_payment_id: string;
  razorpay_signature: string;
}

export interface CartItemData {
  id: string;
  cart_item_id: string;
  product_id: string;
  variant_id: string;
  title: string;
  price: number;
  color?: string;
  size?: string;
  quantity: number;
  line_total: number;
  in_stock: boolean;
  image_url?: string;
}

export interface CartResponse {
  cart_id: string | null;
  items: CartItemData[];
  subtotal: number;
  item_count: number;
}

export interface PaymentVerificationResponse {
  order_id: string;
  payment_id: string;
  status: string;
  message: string;
}

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: config.apiBaseUrl,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000,
    });

    this.client.interceptors.request.use((requestConfig) => {
      if (config.merchantId) {
        requestConfig.headers['X-Merchant-Id'] = config.merchantId;
      }
      if (config.customerId) {
        requestConfig.headers['X-Customer-Id'] = config.customerId;
      }
      return requestConfig;
    });
  }

  private handleError(error: AxiosError): never {
    if (error.response) {
      const message = (error.response.data as { detail?: string })?.detail || error.message;
      throw new Error(message);
    }
    throw new Error(error.message || 'Network error');
  }

  async agentChat(request: AgentChatRequest): Promise<AgentChatResponse> {
    try {
      const response = await this.client.post<AgentChatResponse>('/api/agent/chat', request);
      return response.data;
    } catch (error) {
      this.handleError(error as AxiosError);
    }
  }

  async getCart(): Promise<CartResponse> {
    try {
      const response = await this.client.get<CartResponse>('/api/cart');
      return response.data;
    } catch (error) {
      this.handleError(error as AxiosError);
    }
  }

  async removeCartItem(cartItemId: string): Promise<{ success: boolean }> {
    try {
      const response = await this.client.delete<{ success: boolean }>(`/api/cart/items/${cartItemId}`);
      return response.data;
    } catch (error) {
      this.handleError(error as AxiosError);
    }
  }

  async checkout(request: CheckoutRequest): Promise<CheckoutResponse> {
    try {
      const response = await this.client.post<CheckoutResponse>('/api/checkout', request);
      return response.data;
    } catch (error) {
      this.handleError(error as AxiosError);
    }
  }

  async verifyPayment(request: PaymentVerificationRequest): Promise<PaymentVerificationResponse> {
    try {
      const response = await this.client.post<PaymentVerificationResponse>('/api/payments/verify', request);
      return response.data;
    } catch (error) {
      this.handleError(error as AxiosError);
    }
  }
}

export const apiClient = new ApiClient();