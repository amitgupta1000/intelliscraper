import React from 'react';

const WelcomeSection: React.FC = () => {
  return (
    <section className="text-center">
      <h1 className="text-3xl font-bold text-gray-900">IntelliScraper</h1>
      <p className="mt-2 text-gray-600 max-w-xl mx-auto">
        Submit a URL and the server will scrape the page (AIOHTTP first, Playwright fallback),
        clean the text, and return neatly formatted Markdown you can download.
      </p>
      <p className="mt-4 text-sm text-gray-500">This front-end is intentionally minimal — paste a public URL to get started.</p>
    </section>
  );
};

export default WelcomeSection;
