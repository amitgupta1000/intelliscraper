/**
 * Represents a single turn in the conversation.
 * 'user' for the user's query, 'assistant' for the AI's response.
 */
export interface ConversationMessage {
  role: 'user' | 'assistant';
  content: string;
}

/**
 * Possible statuses for a research task as returned by the API.
 */
export type ResearchTaskStatus = 'initializing' | 'running' | 'completed' | 'failed' | 'pending' | 'concluded';

/**
 * Represents the state of a research session.
 */
export interface ResearchSession {
  sessionId: string | null;
  conversation: ConversationMessage[];
  isLoading: boolean;
  error: string | null;
  currentStatus: ResearchTaskStatus | 'idle';
  progress: number;
  conclusionMessage: string | null;
  usedCache?: boolean;
  processingUrls?: string[];
}

/**
 * The request body for the /api/research endpoint.
 */
export interface ResearchRequest {
  query: string;
  session_id?: string | null;
  prompt_type?: string;
}

/**
 * The response from the /api/research/{session_id}/result endpoint.
 */
export interface ResearchResultResponse {
  analysis_content: string;
  appendix_content?: string;
  analysis_filename?: string;
  appendix_filename?: string;
  sources?: string[];
  qa_pairs?: Array<{ question: string; answer: string; citations?: any[] }>;
}

/**
 * The response from the /api/research/{session_id}/status endpoint.
 */
export interface ResearchStatusResponse {
  status: ResearchTaskStatus;
  progress: number;
  current_step: string;
  error?: string;
  conclusion_message?: string;
  processing_urls?: string[];
}
