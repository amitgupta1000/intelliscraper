import React, { useState } from 'react';
import { MagnifyingGlassIcon } from '@heroicons/react/24/outline';
import { useResearch } from '../context/ResearchContext';


const ResearchForm: React.FC = () => {
  const { submitQuery, isLoading, currentStatus, clearSession } = useResearch();
  const [query, setQuery] = useState('');
  const isConcluded = currentStatus === 'concluded';

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    await submitQuery(query.trim());
  };

  const exampleQueries = [
    "Tell me about Photon Integrated Circuits chips for AI - how realistic is this??",
    "What was the impact of the Treaty of Versailles on Germany?",
    "Were the Mughals disastrous for India?",
    "What is the outlook for crude oil in 2026"
  ];

  return (
    <div className="space-y-6">
      {/* Header in grey box */}
      <div className="bg-gray-100 rounded-lg p-6 mb-2 text-center space-y-2 border border-gray-200">
        <h2 className="text-2xl font-bold text-gray-900 flex items-center justify-center space-x-3 leading-tight">
          <MagnifyingGlassIcon className="w-6 h-6 text-primary-600" />
          <span>Start Your Research</span>
        </h2>
        <p className="text-gray-600 text-base">
          Enter your research question below and let our AI-powered assistant explore the topic for you.
        </p>
      </div>

      <div className="card p-8 shadow-lg space-y-8">
        <form onSubmit={handleSubmit} className="space-y-8">
          {/* Query Input */}
          <div className="space-y-4">
            <label htmlFor="query" className="block text-lg font-semibold text-gray-800">
              What would you like to research?
            </label>
            <div className="relative">
              <textarea
                id="query"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={isConcluded ? "Research concluded. Start a new session to ask more." : "Enter your research question or topic..."}
                className="input-field h-32 resize-none text-lg"
                required
                disabled={isLoading || isConcluded}
                aria-label="Research question input"
              />
              <div className="absolute bottom-3 right-3 text-sm text-gray-400">
                {query.length}/500
              </div>
            </div>
            
            {/* Example Queries */}
            <div className="space-y-2">
              <p className="text-sm font-medium text-gray-600">Try these examples:</p>
              <div className="flex flex-wrap gap-2">
                {exampleQueries.map((example, index) => (
                  <button
                    key={index}
                    type="button"
                    onClick={() => setQuery(example)}
                    className="text-xs bg-gray-100 hover:bg-primary-50 hover:text-primary-700 px-3 py-1 rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary-400"
                    disabled={isLoading || isConcluded}
                    aria-label={`Use example query: ${example}`}
                  >
                    {example.length > 50 ? `${example.substring(0, 50)}...` : example}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Help Note for Research Type Routing */}
          <div className="bg-blue-50 border-l-4 border-blue-400 p-4 mb-4 rounded">
            <div className="font-semibold text-blue-800 mb-1">Tip: Guide the research type with your query</div>
            <div className="text-blue-700 text-sm">
              <ul className="list-disc pl-5">
                <li>Include the word <b>legal</b> for legal research (e.g., "Legal implications of ...")</li>
                <li>Mention a <b>person's name</b> for person-focused research (e.g., "Biography of Ada Lovelace")</li>
                <li>Use words like <b>share</b>, <b>equity</b>, <b>stock</b> for investment research (e.g., "Outlook for Tesla stock")</li>
                <li>Use words like <b>Macro</b>, <b>equities</b>, <b>bonds</b>, <b>commodities</b> for macro research (e.g., "Macro outlook for gold and bonds")</li>
                <li>Otherwise, general research will be performed</li>
              </ul>
            </div>
          </div>

          {/* Submit Button Only */}
          <div className="flex justify-end pt-4 border-t border-gray-200">
            {isConcluded && (
              <button
                type="button"
                onClick={clearSession}
                className="btn-secondary mr-4 py-3 px-6 text-base font-semibold shadow-lg hover:shadow-xl transition-all duration-200"
              >
                Start New Research
              </button>
            )}
            <button
              type="submit"
              disabled={!query.trim() || isLoading || isConcluded}
              className="btn-primary w-full sm:w-auto py-3 px-6 text-base font-semibold disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-3 shadow-lg hover:shadow-xl transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-primary-400"
              aria-label="Start research"
            >
              <MagnifyingGlassIcon className="w-5 h-5" />
              <span>{isLoading ? 'Researching...' : isConcluded ? 'Research Concluded' : 'Start Research'}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ResearchForm;
