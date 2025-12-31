/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useState, useCallback } from 'react';
import type { ReactNode } from 'react';
import type { ResearchSession } from '../types';
import { startScrape } from './scrapeService';

interface ResearchContextType extends ResearchSession {
  submitQuery: (query: string) => Promise<void>;
  clearSession: () => void;
  concludeSession: (sid?: string, source?: 'manual' | 'auto') => Promise<void>;
}

const ResearchContext = createContext<ResearchContextType | undefined>(undefined);

export const ResearchProvider = ({ children }: { children: ReactNode }) => {
  // Configurable timings

  const [session, setSession] = useState<ResearchSession>({
    sessionId: null,
    conversation: [],
    isLoading: false,
    error: null,
    currentStatus: 'idle',
    progress: 0,
    conclusionMessage: null,
    usedCache: false,
  });

  const clearSession = useCallback(() => {
    setSession({
      sessionId: null,
      conversation: [],
      isLoading: false,
      error: null,
      currentStatus: 'idle',
      progress: 0,
      conclusionMessage: null,
      usedCache: false,
    });
  }, []);

  // concludeSession now accepts optional source: 'manual' | 'auto'
  const concludeSession = useCallback(async (_sid?: string, _source?: 'manual' | 'auto') => {
    // Idempotent conclude: if already concluded, no-op
    setSession(prev => {
      if (prev.currentStatus === 'concluded') return prev;
      return { ...prev, currentStatus: 'concluding', isLoading: true };
    });

    // Determine session id to conclude
    let sidToUse = _sid;
    setSession(prev => {
      if (!sidToUse) sidToUse = prev.sessionId || undefined;
      return prev;
    });

    // Try to call backend endpoint if available
    const API_BASE_URL = (import.meta.env.VITE_API_URL as string) || 'http://localhost:8080';
    if (sidToUse) {
      try {
        await fetch(`${API_BASE_URL}/research/${sidToUse}/conclude`, { method: 'POST' });
      } catch (e) {
        // Non-fatal: proceed to local conclude even if backend call fails
        console.warn('Failed to call backend conclude endpoint', e);
      }
    }

    // Finalize local session state
    setSession(prev => ({
      sessionId: prev.sessionId,
      conversation: prev.conversation,
      isLoading: false,
      error: null,
      currentStatus: 'concluded',
      progress: 100,
      conclusionMessage: 'Concluded',
      usedCache: prev.usedCache,
    }));
    // Clear any attached global scrape result to avoid stale downloads
    try {
      (window as any).__lastScrapeResult = null;
    } catch (e) {
      // ignore
    }
  }, []);

  // Inactivity-based auto-conclude removed to avoid unintended session endings.

  const submitQuery = useCallback(async (query: string) => {
    setSession(prev => ({
      ...prev,
      sessionId: prev.sessionId || Date.now().toString(),
      isLoading: true,
      error: null,
      conversation: [...prev.conversation, { role: 'user', content: query }],
    }));

    try {
      const res = await startScrape(query);
      // append assistant response with full scraped markdown (or text)
      const full = (res.markdown || res.text || '') as string;
      setSession(prev => ({
        ...prev,
        isLoading: false,
        conversation: [...prev.conversation, { role: 'assistant', content: full }],
        currentStatus: 'completed',
        progress: 100,
      }));

      // Optionally attach full result onto window for quick download UI access
      (window as any).__lastScrapeResult = res;
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      setSession(prev => ({
        ...prev,
        isLoading: false,
        error: message || 'An unknown error occurred.',
      }));
    }
  }, []);

  // No unmount cleanup needed for conclude timers (feature removed)

  return (
    <ResearchContext.Provider value={{ ...session, submitQuery, clearSession, concludeSession }}>
      {children}
    </ResearchContext.Provider>
  );
};

export const useResearch = () => {
  const context = useContext(ResearchContext);
  if (context === undefined) {
    throw new Error('useResearch must be used within a ResearchProvider');
  }
  return context;
};