import type {
  ResearchRequest,
  ResearchResultResponse,
  ResearchStatusResponse,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Starts a new research task or sends a follow-up question.
 * @param request - The research request object.
 * @returns The session ID for the ongoing research.
 */
export const startResearch = async (
  request: ResearchRequest
): Promise<string> => {
  const response = await fetch(`${API_BASE_URL}/api/research`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || 'Failed to start research');
  }

  const data = await response.json();
  return data.session_id;
};

/**
 * Polls the status of a research session.
 * @param sessionId - The ID of the session to check.
 * @returns The current status of the research.
 */
export const getResearchStatus = async (
  sessionId: string
): Promise<ResearchStatusResponse> => {
  const response = await fetch(`${API_BASE_URL}/api/research/${sessionId}/status`);

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || 'Failed to get research status');
  }

  return response.json();
};

/**
 * Fetches the final result of a completed research task.
 * @param sessionId - The ID of the session to get results for.
 * @returns The research result.
 */
export const getResearchResult = async (
  sessionId: string
): Promise<ResearchResultResponse> => {
  const response = await fetch(`${API_BASE_URL}/api/research/${sessionId}/result`);

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || 'Failed to get research result');
  }

  return response.json();
};

/**
 * Conclude a research session to prevent further follow-ups.
 * @param sessionId - The ID of the session to conclude.
 */
export const concludeResearch = async (sessionId: string): Promise<{ success: boolean; message: string }> => {
  const response = await fetch(`${API_BASE_URL}/api/research/${sessionId}/conclude`, {
    method: 'POST',
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || 'Failed to conclude research session');
  }

  return response.json();
};
