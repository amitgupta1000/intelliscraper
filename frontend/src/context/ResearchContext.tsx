/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useState, useCallback } from 'react';
import type { ReactNode } from 'react';
import type { ResearchSession, ResearchRequest, ResearchStatusResponse, ResearchResultResponse } from '../types';
import {
  startResearch,
  getResearchStatus,
  getResearchResult,
  concludeResearch,
} from './researchService';

interface ResearchContextType extends ResearchSession {
  submitQuery: (query: string) => Promise<void>;
  clearSession: () => void;
  concludeSession: (sid?: string, source?: 'manual' | 'auto') => Promise<void>;
}

const ResearchContext = createContext<ResearchContextType | undefined>(undefined);

export const ResearchProvider = ({ children }: { children: ReactNode }) => {
  // Configurable timings
  const POLL_INTERVAL_MS = 3 * 1000; // poll backend every 3s (was 1s)

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
  const concludeSession = useCallback(async (sid?: string) => {
    const id = sid || session.sessionId;
    if (!id) return;
    setSession(prev => ({ ...prev, isLoading: true }));
    try {
      const res = await concludeResearch(id);
      const message = (res && res.message) || 'Research concluded';
      setSession(prev => ({
        ...prev,
        isLoading: false,
        currentStatus: 'concluded',
        conclusionMessage: message,
        progress: 100,
      }));
      // no-op: we don't show auto-conclude toasts
    } catch (err: unknown) {
      // If the backend conclude fails, mark locally concluded to avoid hanging sessions
      const message = err instanceof Error ? err.message : String(err);
      setSession(prev => ({
        ...prev,
        isLoading: false,
        currentStatus: 'concluded',
        conclusionMessage: 'Research concluded (local)',
        progress: 100,
        error: message || null,
      }));
      // no-op: avoid auto-conclude toast
    }

    // Immediately reset the UI session state so the app returns to idle.
    try {
      clearSession();
    } catch (e) {
      console.error('clearSession failed after conclude', e);
    }
  }, [session.sessionId, clearSession]);

  // Inactivity-based auto-conclude removed to avoid unintended session endings.

  const submitQuery = useCallback(async (query: string) => {
    // Preserve existing sessionId if present; do not clear session state.
    setSession(prev => ({
      ...prev,
      isLoading: true,
      error: null,
      conversation: [...prev.conversation, { role: 'user', content: query }],
    }));

      try {
      const request: ResearchRequest = {
        query,
        prompt_type: 'general',
        session_id: session.sessionId || undefined,
      };

      const newSessionId = await startResearch(request);

      setSession(prev => ({ ...prev, sessionId: newSessionId }));

      const poll = async () => {
        try {
          const status: ResearchStatusResponse = await getResearchStatus(newSessionId);
          setSession(prev => ({
            ...prev,
            currentStatus: status.current_step as ResearchSession['currentStatus'],
            progress: status.progress,
            processingUrls: status.processing_urls || [],
          }));

          // Handle concluded sessions explicitly
          if (status.status === 'concluded') {
            setSession(prev => ({
              ...prev,
              isLoading: false,
              currentStatus: 'concluded',
              conclusionMessage: status.conclusion_message || null,
              progress: status.progress || 100,
              usedCache: status.used_cache || false,
            }));
            return;
          }

            if (status.status === 'completed') {
            const result: ResearchResultResponse = await getResearchResult(newSessionId);
            setSession(prev => ({
              ...prev,
              isLoading: false,
              conversation: [
                ...prev.conversation,
                { role: 'assistant', content: result.analysis_content },
              ],
              progress: 100,
              currentStatus: 'completed',
              usedCache: status.used_cache || false,
            }));

            // If we received a non-empty analysis result, leave the session in
            // `completed` state and allow the UI to display results. The user
            // can choose to download the analysis via the Results UI button.
          } else if (status.status === 'failed') {
            throw new Error(status.detail || 'Research pipeline failed.');
          
          } else {
            setTimeout(poll, POLL_INTERVAL_MS);
          }
        } catch (err: unknown) {
          const message = err instanceof Error ? err.message : String(err);
          setSession(prev => ({
            ...prev,
            isLoading: false,
            error: message || 'An unknown polling error occurred.',
          }));
        }
      };

      // NOTE: Inactivity-based auto-conclude removed; we only download results
      // immediately when available and leave the session in `completed` state.

      poll();

    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      setSession(prev => ({
        ...prev,
        isLoading: false,
        error: message || 'An unknown error occurred.',
      }));
    }
  }, [session.sessionId, POLL_INTERVAL_MS]);

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