export const config = {
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  merchantId: import.meta.env.VITE_MERCHANT_ID || '',
  customerId: import.meta.env.VITE_CUSTOMER_ID || '',
} as const;

export type Config = typeof config;