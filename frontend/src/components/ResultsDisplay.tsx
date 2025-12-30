import { useState } from 'react';
import { useResearch } from '../context/ResearchContext';
import ReactMarkdown from 'react-markdown';
import ProgressBar from './ProgressBar';
import ResearchInProgress from './ResearchInProgress';

export const ResultsDisplay = () => {
  const { 
    conversation, 
    isLoading, 
    currentStatus, 
    progress, 
    error,  
    conclusionMessage,
    usedCache
  } = useResearch();
  const [isConcluding, setIsConcluding] = useState(false);

  // Find the latest assistant message (the analytical result)
  const lastAssistantMsg = [...conversation].reverse().find((msg) => msg.role === 'assistant');
  // Use the most recent user query (in case conversation contains previous sessions)
  const userQuery = [...conversation].reverse().find((msg) => msg.role === 'user');

  // Handler to let the user download the analysis (does NOT conclude)
  const downloadOnly = () => {
    if (!lastAssistantMsg) return;
    try {
      setIsConcluding(true);

      const parts: string[] = [];
      if (userQuery) parts.push(`Research Question:\n${userQuery.content}\n\n`);
      parts.push('Analysis:\n');
      parts.push(lastAssistantMsg.content);

      const blob = new Blob([parts.join('')], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const filename = `analysis_${new Date().toISOString().replace(/[:.]/g, '-')}.txt`;

      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);

      // Small delay to allow download to start
      setTimeout(() => {
        setIsConcluding(false);
      }, 500);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error('Download failed', err);
      setIsConcluding(false);
    }
  };

  // Auto-conclude is handled centrally by `ResearchContext` if needed.

  // Download conversation as text file
  // Saving and concluding are now automatic when results arrive.

  return (
    <div className="results-frame max-h-[600px] overflow-y-auto p-6 bg-white rounded-lg shadow-md space-y-6">
      {isLoading ? (
        <div className="mb-6">
          <ResearchInProgress />
          <div className="mt-6">
            <ProgressBar progress={progress} />
            <div className="text-center text-gray-600 mt-2">{currentStatus}</div>
          </div>
        </div>
      ) : lastAssistantMsg ? (
        <div className="analytical-result space-y-6">
          {/* Conclusion Message */}
          {conclusionMessage && (
            <div className="bg-green-50 border-l-4 border-green-500 p-4 mb-4">
              <p className="text-green-700 font-medium">{conclusionMessage}</p>
            </div>
          )}

          {/* Cache badge */}
          {usedCache && (
            <div className="text-sm text-gray-600 italic">Results served from cache</div>
          )}

          {/* User Query removed — question is included in the report itself */}

          {/* Analytical Sections (parse markdown for sections) */}
          <div className="section-box bg-white p-4 rounded border">
            <ReactMarkdown
              components={{
                h1: ({node, ...props}) => <h2 className="font-bold text-xl mt-4 mb-2 text-primary-700" {...props} />,
                h2: ({node, ...props}) => <h3 className="font-semibold text-lg mt-3 mb-1 text-primary-600" {...props} />,
                ul: ({node, ...props}) => <ul className="list-disc ml-6 space-y-1" {...props} />,
                li: ({node, ...props}) => <li className="text-gray-800" {...props} />,
                strong: ({node, ...props}) => <strong className="font-bold text-primary-700" {...props} />,
                p: ({node, ...props}) => <p className="mb-2 text-gray-900" {...props} />,
                blockquote: ({node, ...props}) => <blockquote className="border-l-4 pl-4 italic text-gray-600 my-2" {...props} />,
                code: ({node, ...props}) => <code className="bg-gray-100 px-1 rounded text-sm" {...props} />,
              }}
            >
              {lastAssistantMsg.content}
            </ReactMarkdown>
          </div>

          {/* Conclusion Section (if present in markdown, highlight) */}
          {lastAssistantMsg.content.toLowerCase().includes('conclusion') && (
            <div className="conclusion-box bg-primary-50 p-4 rounded border border-primary-200 mt-4">
              <div className="font-bold text-primary-700 mb-1">Conclusion</div>
              <ReactMarkdown>{
                lastAssistantMsg.content.split(/##?\s*Conclusion/i)[1] || ''
              }</ReactMarkdown>
            </div>
          )}

          {/* Session conclusion and save happen automatically; show status */}
          <div className="mt-6">
            {currentStatus === 'concluded' ? (
              <div className="inline-block bg-green-50 border-l-4 border-green-500 p-3">
                <span className="text-green-700 font-medium">Research concluded and saved.</span>
              </div>
            ) : (
              <div className="text-sm text-gray-600">Results available — finalizing and saving...</div>
            )}
            <div className="mt-3">
              <button
                onClick={downloadOnly}
                disabled={isConcluding}
                className="inline-flex items-center px-3 py-2 bg-primary-600 text-white text-sm rounded hover:bg-primary-700 disabled:opacity-60"
              >
                {isConcluding ? 'Preparing download...' : 'Download results'}
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div className="text-gray-500 text-center py-12">No research results to display yet.</div>
      )}

      {/* Error Message */}
      {error && <div className="error-message text-red-600 mt-4 text-center">{error}</div>}
    </div>
  );
};