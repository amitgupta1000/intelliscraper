import React, { useState } from 'react';
import { useResearch } from '../context/ResearchContext';

export const RequestForm = () => {
  const [url, setUrl] = useState('');
  const { submitQuery, isLoading, clearSession, currentStatus } = useResearch();
  const isConcluded = currentStatus === 'concluded';

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = url.trim();
    if (!trimmed || isLoading || isConcluded) return;
    // submitQuery expects a string — we'll send the URL
    submitQuery(trimmed);
    setUrl('');
  };

  return (
    <div className="bg-white p-6 rounded shadow">
      <h2 className="text-lg font-semibold mb-3">Enter a URL to scrape</h2>
      <form onSubmit={handleSubmit} className="flex gap-3">
        <input
          type="url"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder={isConcluded ? 'Session concluded' : 'https://example.com/article'}
          disabled={isLoading || isConcluded}
          className="flex-1 input-field px-3 py-2 border rounded"
          required
          aria-label="URL to scrape"
        />
        <button
          type="submit"
          disabled={!url.trim() || isLoading || isConcluded}
          className="px-4 py-2 bg-primary-600 text-white rounded disabled:opacity-60"
        >
          {isLoading ? 'Processing...' : 'Scrape'}
        </button>
      </form>

      <div className="mt-3 text-sm text-gray-600">
        Paste a public URL. The app will try AIOHTTP first, then Playwright as a fallback.
      </div>

      {isConcluded && (
        <div className="mt-3">
          <button onClick={clearSession} className="px-3 py-1 bg-gray-100 rounded">Start new session</button>
        </div>
      )}
    </div>
  );
};