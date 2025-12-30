import React from 'react';
import { CpuChipIcon, ArrowTopRightOnSquareIcon, HeartIcon } from '@heroicons/react/24/outline';

const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-white border-t border-gray-200 mt-16">
      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Brand Section */}
          <div className="col-span-1 md:col-span-2">
            <div className="flex items-center space-x-3 mb-4">
              <div className="flex items-center justify-center w-10 h-10 bg-primary-600 rounded-lg">
                <CpuChipIcon className="w-6 h-6 text-white" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-gray-900">INTELLISEARCH</h3>
                <p className="text-sm text-gray-600">AI-Powered Research Assistant</p>
              </div>
            </div>
            <p className="text-gray-600 text-sm leading-relaxed max-w-md">
              Transform your research process with advanced AI technology. Get comprehensive, 
              well-sourced reports on any topic with the power of Google Gemini AI and 
              intelligent web scraping.
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="text-sm font-semibold text-gray-900 mb-4">Features</h4>
            <ul className="space-y-2 text-sm text-gray-600">
              <li>AI-Powered Analysis</li>
              <li>Comprehensive Reports</li>
              <li>Source Citations</li>
              <li>Real-time Progress</li>
              <li>Multiple Report Types</li>
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h4 className="text-sm font-semibold text-gray-900 mb-4">Resources</h4>
            <ul className="space-y-2">
              <li>
                <a 
                  href="/api/docs" 
                  target="_blank"
                  className="text-sm text-gray-600 hover:text-primary-600 transition-colors flex items-center space-x-1"
                >
                  <span>API Documentation</span>
                  <ArrowTopRightOnSquareIcon className="w-3 h-3" />
                </a>
              </li>
              <li>
                <a 
                  href="https://ArrowTopRightOnSquareIcon.com" 
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-gray-600 hover:text-primary-600 transition-colors flex items-center space-x-1"
                >
                  <ArrowTopRightOnSquareIcon className="w-3 h-3" />
                  <span>Source Code</span>
                </a>
              </li>
              <li>
                <a 
                  href="#" 
                  className="text-sm text-gray-600 hover:text-primary-600 transition-colors"
                >
                  Research Tips
                </a>
              </li>
              <li>
                <a 
                  href="#" 
                  className="text-sm text-gray-600 hover:text-primary-600 transition-colors"
                >
                  User Guide
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom Section */}
        <div className="border-t border-gray-200 mt-8 pt-6">
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            <div className="flex items-center space-x-4 text-sm text-gray-600">
              <span>© {currentYear} INTELLISEARCH. All rights reserved.</span>
            </div>
            
            <div className="flex items-center space-x-6 text-sm text-gray-600">
              <span className="flex items-center space-x-1">
                <span>Made with</span>
                <HeartIcon className="w-4 h-4 text-red-500" />
                <span>using React & FastAPI</span>
              </span>
            </div>
          </div>
          
          {/* Tech Stack */}
          <div className="mt-4 pt-4 border-t border-gray-100">
            <div className="flex flex-wrap justify-center gap-4 text-xs text-gray-500">
              <span className="bg-gray-100 px-2 py-1 rounded">Google Gemini AI</span>
              <span className="bg-gray-100 px-2 py-1 rounded">React 19</span>
              <span className="bg-gray-100 px-2 py-1 rounded">TypeScript</span>
              <span className="bg-gray-100 px-2 py-1 rounded">FastAPI</span>
              <span className="bg-gray-100 px-2 py-1 rounded">Tailwind CSS</span>
              <span className="bg-gray-100 px-2 py-1 rounded">LangGraph</span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
