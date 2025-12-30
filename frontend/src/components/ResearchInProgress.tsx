import React from 'react';
import { BeakerIcon, MagnifyingGlassIcon, DocumentTextIcon } from '@heroicons/react/24/outline';
import { useResearch } from '../context/ResearchContext';

const ResearchInProgress: React.FC = () => {
  const { processingUrls, currentStatus } = useResearch();

  // Prefer explicit backend step names, but fall back to a gentle heuristic
  const writingSteps = new Set(['writing_report', 'generating_report', 'synthesizing_report', 'writing_analysis']);
  const normStep = (currentStatus || '').toString().toLowerCase();
  const isWriting = writingSteps.has(normStep) || /report|write|generat|synthesiz/i.test(normStep);

  return (
    <div className="card p-8 shadow-lg text-center">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Research in Progress...</h2>
      <div className="flex justify-center items-start space-x-8">
        <div className="flex flex-col items-center animate-float animation-delay-100">
          <BeakerIcon className="w-12 h-12 text-primary-600" />
          <p className="mt-2 text-sm text-gray-600">Formulating strategy</p>
        </div>
        <div className="flex-1 text-left">
          <div className="flex items-center space-x-3">
            <MagnifyingGlassIcon className="w-10 h-10 text-primary-600" />
            <div>
              <p className="text-sm font-semibold">Gathering sources</p>
              <p className="text-xs text-gray-500">Processing the following URLs:</p>
            </div>
          </div>
          <div className="mt-3 max-h-40 overflow-y-auto text-left p-2 bg-gray-50 rounded border border-gray-100">
            {processingUrls && processingUrls.length > 0 ? (
              <ul className="text-sm space-y-2">
                {processingUrls.map((u, i) => (
                  <li key={i} className="flex items-center space-x-2">
                    <svg className="w-4 h-4 animate-spin text-primary-600" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"></path>
                    </svg>
                    <span className="truncate">{u}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-gray-400">Discovering sources...</p>
            )}
          </div>
        </div>
        <div className="flex flex-col items-center animate-float animation-delay-300">
          <DocumentTextIcon className={`w-12 h-12 ${isWriting ? 'text-primary-700' : 'text-primary-400'}`} />
          <p className={`mt-2 text-sm ${isWriting ? 'text-gray-900' : 'text-gray-600'}`}>{isWriting ? 'Generating report' : 'Preparing report'}</p>
        </div>
      </div>
      <div className="mt-8">
        <div className="w-full bg-gray-200 rounded-full h-2.5">
          <div className="bg-primary-600 h-2.5 rounded-full animate-pulse-glow"></div>
        </div>
        <p className="text-sm text-gray-500 mt-2">This may take a minute. Please don't close this window.</p>
      </div>
    </div>
  );
};

export default ResearchInProgress;
