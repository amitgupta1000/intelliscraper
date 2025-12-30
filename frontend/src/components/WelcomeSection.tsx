import React from 'react';
import { CpuChipIcon, BoltIcon, DocumentTextIcon, UsersIcon, GlobeAltIcon, TagIcon } from '@heroicons/react/24/outline';

const WelcomeSection: React.FC = () => {
  const features = [
    {
      icon: <CpuChipIcon className="w-6 h-6" />,
      title: "AI-Powered Research",
      description: "Advanced Web-enabled Crystal AI analyzes and synthesizes information"
    },
    {
      icon: <BoltIcon className="w-6 h-6" />,
      title: "Lightning Fast",
      description: "Get comprehensive research reports in minutes, not hours"
    },
    {
      icon: <DocumentTextIcon className="w-6 h-6" />,
      title: "Professional Reports",
      description: "Structured reports with citations and source validation"
    },
    {
      icon: <GlobeAltIcon className="w-6 h-6" />,
      title: "Customisable Search",
      description: "Select between Fast and Ultra scans to suit your needs"
    }
  ];

  return (
    <section className="text-center space-y-8">
      <div className="space-y-4">
        <h1 className="text-4xl md:text-6xl font-bold text-gray-900">
          Welcome to{' '}
          <span className="bg-gradient-to-r from-primary-600 to-blue-600 bg-clip-text text-transparent">
            INTELLISEARCH
          </span>
        </h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
          Transform your research process with AI-powered intelligence. Get comprehensive, 
          well-sourced reports on any topic in minutes.
        </p>
      </div>

      <div className="flex flex-wrap justify-center gap-4">
        <div className="flex items-center space-x-2 bg-white px-4 py-2 rounded-full shadow-sm border">
          <TagIcon className="w-4 h-4 text-green-500" />
          <span className="text-sm font-medium text-gray-700">Accurate Research</span>
        </div>
        <div className="flex items-center space-x-2 bg-white px-4 py-2 rounded-full shadow-sm border">
          <UsersIcon className="w-4 h-4 text-blue-500" />
          <span className="text-sm font-medium text-gray-700">Trusted by Researchers</span>
        </div>
        <div className="flex items-center space-x-2 bg-white px-4 py-2 rounded-full shadow-sm border">
          <BoltIcon className="w-4 h-4 text-yellow-500" />
          <span className="text-sm font-medium text-gray-700">Instant Results</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-12">
        {features.map((feature, index) => (
          <div key={index} className="card p-6 text-center hover:shadow-lg transition-shadow duration-200">
            <div className="flex justify-center mb-4">
              <div className="p-3 bg-primary-100 text-primary-600 rounded-lg">
                {feature.icon}
              </div>
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">{feature.title}</h3>
            <p className="text-gray-600 text-sm">{feature.description}</p>
          </div>
        ))}
      </div>
    </section>
  );
};

export default WelcomeSection;
