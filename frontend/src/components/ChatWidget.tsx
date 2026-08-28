import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import type { ChatMessage as ChatMessageType, AgentProductSummary } from '../types';

interface ChatWidgetProps {
  messages: ChatMessageType[];
  isLoading: boolean;
  error: string | null;
  onSendMessage: (message: string) => void;
  onAddToCart: (product: AgentProductSummary) => void;
  onBuyNow: (product: AgentProductSummary) => void;
  messagesEndRef: React.RefObject<HTMLDivElement>;
}

export function ChatWidget({
  messages,
  isLoading,
  error,
  onSendMessage,
  onAddToCart,
  onBuyNow,
  messagesEndRef,
}: ChatWidgetProps) {
  return (
    <div className="chat-widget">
      <div className="chat-header">
        <h2>AI Shopping Assistant</h2>
        {error && <div className="chat-error">{error}</div>}
      </div>

      <div className="chat-messages">
        {messages.map((message) => (
          <ChatMessage
            key={message.id}
            message={message}
            onAddToCart={onAddToCart}
            onBuyNow={onBuyNow}
          />
        ))}
        <div ref={messagesEndRef} />
      </div>

      {isLoading && (
        <div className="chat-loading">
          <span className="loading-dots">
            <span>.</span><span>.</span><span>.</span>
          </span>
          <span>AI is thinking...</span>
        </div>
      )}

      <ChatInput onSendMessage={onSendMessage} disabled={isLoading} />
    </div>
  );
}