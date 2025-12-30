

import Header from './components/Header';
import Footer from './components/Footer';
import WelcomeSection from './components/WelcomeSection';
import ResearchForm from './components/ResearchForm';
import { ResultsDisplay } from './components/ResultsDisplay';
import { useResearch, ResearchProvider } from './context/ResearchContext';

const MainContent = () => {
  const { sessionId, conversation } = useResearch();
  return (
    <>
      <WelcomeSection />
      <div className="w-full max-w-7xl mx-auto mt-10 px-2 md:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
          {/* Main research form */}
          <div className="col-span-2 space-y-8">
            <ResearchForm />
          </div>
          {/* Results in the second column (sidebar) */}
          <aside className="col-span-1 sticky top-28 self-start space-y-8">
            {/* Grey box header for Research Results */}
            <div className="bg-gray-100 rounded-lg p-6 mb-2 text-center border border-gray-200">
              <h2 className="text-2xl font-bold text-gray-900">Research Results</h2>
              <p className="text-gray-600 text-sm mt-1">Your analytical results and conclusions will appear here.</p>
            </div>
            {sessionId && conversation.length > 0 && (
              <ResultsDisplay />
            )}
            <div className="card p-6 mb-6">
              <h3 className="font-bold text-lg mb-2 text-primary-700">Session Summary</h3>
              <ul className="list-disc ml-5 text-gray-700 text-sm space-y-1">
                <li>Research is private and session-based</li>
                <li>Ask follow-up questions for deeper analysis</li>
                <li>Download your results anytime</li>
                <li>Progress and status are always visible</li>
              </ul>
            </div>
            {/* Placeholder for future widgets (sources, tips, etc.) */}
            <div className="card p-4 text-sm text-gray-500">More features coming soon...</div>
          </aside>
        </div>
      </div>
    </>
  );
};

function App() {
  return (
    <ResearchProvider>
      <div className="app-container min-h-screen flex flex-col">
        <Header />
        <main className="flex-1">
          <MainContent />
        </main>
        <Footer />
      </div>
    </ResearchProvider>
  );
}

export default App;
