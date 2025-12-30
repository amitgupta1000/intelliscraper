import React, { useState } from 'react';
import { useResearch } from '../context/ResearchContext';

export const RequestForm = () => {
  const [query, setQuery] = useState('');
  const { submitQuery, isLoading, clearSession, sessionId, currentStatus } = useResearch();
  const isConcluded = currentStatus === 'concluded';

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      submitQuery(query);
      setQuery('');
    }
  };

  return (
    <div className="request-form-container">
      <form onSubmit={handleSubmit} className="request-form">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={
            isConcluded 
              ? "Research concluded" 
              : sessionId 
                ? "Ask a follow-up..." 
                : "Start your research..."
          }
          disabled={isLoading || isConcluded}
        />
        <button type="submit" disabled={isLoading || !query.trim() || isConcluded}>
          {isLoading ? 'Thinking...' : 'Send'}
        </button>
      </form>
      {isConcluded && sessionId && (
        <button onClick={clearSession} disabled={isLoading} className="new-session-btn">
          Start New Research
        </button>
      )}
    </div>
  );
};