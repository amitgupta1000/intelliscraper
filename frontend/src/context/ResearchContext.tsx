/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useState, useCallback } from 'react';
import type { ResearchSession } from '../types';
import type { ReactNode } from 'react';
import { startScrape } from './scrapeService';


interface ResearchContextValue extends ResearchSession {
  submitQuery: (query: string) => Promise<void>;
  clearSession: () => void;
  concludeSession: (sid?: string) => Promise<void>;
}

const ResearchContext = createContext<ResearchContextValue | undefined>(undefined);

const initialSessionState: ResearchSession = {
  sessionId: null,
  conversation: [],
  isLoading: false,
  error: null,
  currentStatus: 'idle',
  progress: 0,
  conclusionMessage: null,
  usedCache: false,
};

export const ResearchProvider = ({ children }: { children: ReactNode }) => {
  const [session, setSession] = useState<ResearchSession>(initialSessionState);

  const clearSession = useCallback(() => {
    setSession(initialSessionState);
  }, []);

  const concludeSession = useCallback(async (sid?: string) => {
    setSession(prev => {
      if (prev.currentStatus === 'concluded') return prev;
      return { ...prev, currentStatus: 'concluding', isLoading: true };
    });

    const sidToUse = sid ?? session.sessionId ?? undefined;
    const API_BASE_URL = import.meta.env.VITE_API_URL as string || 'http://localhost:8080';

    if (sidToUse) {
      try {
        await fetch(`${API_BASE_URL}/research/${sidToUse}/conclude`, { method: 'POST' });
      } catch (e) {
        console.warn('Failed to call backend conclude endpoint', e);
      }
    }

    setSession(prev => ({
      ...prev,
      isLoading: false,
      error: null,
      currentStatus: 'concluded',
      progress: 100,
      conclusionMessage: 'Concluded',
    }));

    (window as any).__lastScrapeResult = null;
  }, [session.sessionId]);

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
      const full = res.markdown ?? res.text ?? '';
      setSession(prev => ({
        ...prev,
        isLoading: false,
        conversation: [...prev.conversation, { role: 'assistant', content: full }],
        currentStatus: 'completed',
        progress: 100,
      }));
      (window as any).__lastScrapeResult = res;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setSession(prev => ({
        ...prev,
        isLoading: false,
        error: message || 'An unknown error occurred.',
      }));
    }
  }, []);

  return (
    <ResearchContext.Provider value={{ ...session, submitQuery, clearSession, concludeSession }}>
      {children}
    </ResearchContext.Provider>
  );
};

export const useResearch = () => {
  const context = useContext(ResearchContext);
  if (!context) {
    throw new Error('useResearch must be used within a ResearchProvider');
  }
  return context;
};