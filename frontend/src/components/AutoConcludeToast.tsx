import React from 'react';
import { useResearch } from '../context/ResearchContext';

const AutoConcludeToast: React.FC = () => {
  const { autoConcludeToast } = useResearch() as any;

  if (!autoConcludeToast) return null;

  return (
    <div className="fixed bottom-6 right-6 z-50">
      <div className="bg-primary-700 text-white px-4 py-2 rounded shadow-lg">
        Research session auto-concluded due to inactivity.
      </div>
    </div>
  );
};

export default AutoConcludeToast;
