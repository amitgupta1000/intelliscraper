

import WelcomeSection from './components/WelcomeSection';
import { RequestForm } from './components/RequestForm';
import { ResultsDisplay } from './components/ResultsDisplay';
import LoadingSpinner from './components/LoadingSpinner';
import { useResearch, ResearchProvider } from './context/ResearchContext';

const MainContent = () => {
  const { sessionId, conversation, isLoading } = useResearch();
  const hasResults = sessionId && conversation.length > 0;

  return (
    <div className="w-full max-w-3xl mx-auto mt-12 px-4">
      <WelcomeSection />

      <div className="mt-8">
        <RequestForm />
      </div>

      <div className="mt-6">
        {isLoading && (
          <div className="p-6 bg-white rounded shadow text-center">
            <div className="mb-3">Processing — fetching and cleaning content...</div>
            <div className="flex justify-center">
              <LoadingSpinner size="lg" />
            </div>
          </div>
        )}

        {hasResults && (
          <div className="mt-6">
            <ResultsDisplay />
          </div>
        )}
      </div>
    </div>
  );
};

function App() {
  return (
    <ResearchProvider>
      <div className="min-h-screen bg-gray-50">
        <main>
          <MainContent />
        </main>
      </div>
    </ResearchProvider>
  );
}

export default App;
