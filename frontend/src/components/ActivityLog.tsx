import React from 'react';
import { useResearch } from '../context/ResearchContext';
import {
  ChatBubbleLeftRightIcon,
  UserIcon,
  PaperAirplaneIcon,
} from '@heroicons/react/24/outline';

const ActivityLog: React.FC = () => {
  const { conversation, isLoading, currentStatus } = useResearch();

  return (
    <div className="mt-6 bg-white rounded-lg shadow-md border border-gray-200">
      <div className="px-6 py-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 flex items-center space-x-2">
          <ChatBubbleLeftRightIcon className="w-5 h-5" />
          <span>Conversation Log</span>
        </h3>
        {isLoading && (
          <p className="text-sm text-gray-600 mt-1">{currentStatus}</p>
        )}
      </div>

      <div className="max-h-96 overflow-y-auto p-4 space-y-4">
        {conversation.length === 0 && !isLoading ? (
          <div className="px-6 py-4 text-gray-500 text-center">
            No conversation yet. Start your research to see the exchange.
          </div>
        ) : (
          conversation.map((message, index) => (
            <div
              key={index}
              className={`flex items-start space-x-3 ${
                message.role === 'user' ? 'justify-end' : ''
              }`}
            >
              {message.role === 'assistant' && (
                <div className="w-8 h-8 rounded-full bg-primary-100 flex items-center justify-center border border-primary-200">
                  <PaperAirplaneIcon className="w-5 h-5 text-primary-600" />
                </div>
              )}
              <div
                className={`p-3 rounded-lg max-w-lg ${
                  message.role === 'user'
                    ? 'bg-blue-50 border border-blue-200 text-blue-900'
                    : 'bg-gray-50 border border-gray-200 text-gray-900'
                }`}
              >
                <p className="text-sm">{message.content}</p>
              </div>
              {message.role === 'user' && (
                <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center border border-blue-200">
                  <UserIcon className="w-5 h-5 text-blue-600" />
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default ActivityLog;