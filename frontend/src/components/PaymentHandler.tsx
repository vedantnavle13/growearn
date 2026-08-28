import { useEffect, useRef, useCallback } from 'react';
import type { PaymentState } from '../types';

interface PaymentHandlerProps {
  state: PaymentState;
  onComplete: (paymentData: {
    razorpay_payment_id: string;
    razorpay_order_id: string;
    razorpay_signature: string;
  }) => void;
  onClose: () => void;
}

declare global {
  interface Window {
    Razorpay: new (options: RazorpayOptions) => RazorpayInstance;
  }
}

interface RazorpayOptions {
  key: string;
  amount: number;
  currency: string;
  order_id: string;
  name: string;
  description: string;
  handler: (response: RazorpayResponse) => void;
  prefill?: {
    name?: string;
    email?: string;
    contact?: string;
  };
  theme?: {
    color?: string;
  };
  modal?: {
    ondismiss?: () => void;
  };
}

interface RazorpayResponse {
  razorpay_payment_id: string;
  razorpay_order_id: string;
  razorpay_signature: string;
}

interface RazorpayInstance {
  open: () => void;
  close: () => void;
}

export function PaymentHandler({ state, onComplete, onClose }: PaymentHandlerProps) {
  const razorpayLoaded = useRef(false);

  const loadRazorpayScript = useCallback((): Promise<void> => {
    return new Promise((resolve, reject) => {
      if (window.Razorpay) {
        resolve();
        return;
      }

      const script = document.createElement('script');
      script.src = 'https://checkout.razorpay.com/v1/checkout.js';
      script.async = true;
      script.onload = () => resolve();
      script.onerror = () => reject(new Error('Failed to load Razorpay script'));
      document.body.appendChild(script);
    });
  }, []);

  const openRazorpayCheckout = useCallback(() => {
    if (!state.keyId || !state.razorpayOrderId || !state.amount) return;

    const options: RazorpayOptions = {
      key: state.keyId,
      amount: state.amount,
      currency: state.currency || 'INR',
      order_id: state.razorpayOrderId,
      name: 'Growearn',
      description: 'Order Payment',
      handler: (response: RazorpayResponse) => {
        onComplete({
          razorpay_payment_id: response.razorpay_payment_id,
          razorpay_order_id: response.razorpay_order_id,
          razorpay_signature: response.razorpay_signature,
        });
      },
      prefill: {
        name: 'Customer',
        email: 'customer@example.com',
        contact: '9999999999',
      },
      theme: {
        color: '#3399cc',
      },
      modal: {
        ondismiss: () => {
          if (state.status !== 'success' && state.status !== 'failure') {
            onClose();
          }
        },
      },
    };

    try {
      const rzp = new window.Razorpay(options);
      rzp.open();
    } catch (error) {
      console.error('Razorpay initialization failed:', error);
      onClose();
    }
  }, [state.keyId, state.razorpayOrderId, state.amount, state.currency, state.status, onComplete, onClose]);

  useEffect(() => {
    if (state.status === 'payment_pending' && state.keyId && state.razorpayOrderId && state.amount) {
      loadRazorpayScript().then(() => {
        if (window.Razorpay) {
          razorpayLoaded.current = true;
          openRazorpayCheckout();
        }
      });
    }

    return () => {
      if (razorpayLoaded.current && window.Razorpay) {
        // Cleanup if needed
      }
    };
  }, [state.status, state.keyId, state.razorpayOrderId, state.amount, loadRazorpayScript, openRazorpayCheckout]);

  if (state.status === 'loading') {
    return (
      <div className="payment-overlay">
        <div className="payment-modal">
          <div className="payment-loading">
            <div className="spinner"></div>
            <p>Initializing payment...</p>
          </div>
        </div>
      </div>
    );
  }

  if (state.status === 'payment_processing') {
    return (
      <div className="payment-overlay">
        <div className="payment-modal">
          <div className="payment-loading">
            <div className="spinner"></div>
            <p>Verifying payment...</p>
          </div>
        </div>
      </div>
    );
  }

  if (state.status === 'failure') {
    return (
      <div className="payment-overlay">
        <div className="payment-modal error">
          <h3>Payment Failed</h3>
          <p>{state.error || 'An error occurred during payment'}</p>
          <button className="btn-retry" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    );
  }

  return null;
}