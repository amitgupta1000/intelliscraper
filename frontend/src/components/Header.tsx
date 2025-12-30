import React, { useState } from 'react';
import { CpuChipIcon, Bars3Icon, XMarkIcon, Cog6ToothIcon } from '@heroicons/react/24/outline';

const Header: React.FC = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  return (
    <header className="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo and Brand */}
          <div className="flex items-center space-x-3">
            <div className="flex items-center justify-center w-10 h-10 bg-gradient-to-r from-primary-600 to-blue-600 rounded-lg">
              <CpuChipIcon className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">INTELLISEARCH</h1>
              <p className="text-xs text-gray-600 hidden sm:block">AI-Powered Research Assistant</p>
            </div>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-6">
            <nav className="flex items-center space-x-4">
              <a href="#" className="text-sm text-gray-600 hover:text-primary-600 transition-colors">
                Home
              </a>
              <a href="/api/docs" target="_blank" className="text-sm text-gray-600 hover:text-primary-600 transition-colors">
                API Docs
              </a>
              <a href="#" className="text-sm text-gray-600 hover:text-primary-600 transition-colors">
                Help
              </a>
            </nav>
            
            <div className="flex items-center space-x-3 pl-4 border-l border-gray-200">
              <div className="flex items-center space-x-2 text-sm">
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-gray-600">AI Ready</span>
                </div>
              </div>
              
              <button className="p-2 text-gray-400 hover:text-gray-600 transition-colors">
                <Cog6ToothIcon className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden">
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
            >
              {isMenuOpen ? <XMarkIcon className="w-6 h-6" /> : <Bars3Icon className="w-6 h-6" />}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        {isMenuOpen && (
          <div className="md:hidden border-t border-gray-200 bg-white">
            <div className="px-4 py-4 space-y-4">
              <nav className="space-y-2">
                <a href="#" className="block text-sm text-gray-600 hover:text-primary-600 transition-colors py-2">
                  Home
                </a>
                <a href="/api/docs" target="_blank" className="block text-sm text-gray-600 hover:text-primary-600 transition-colors py-2">
                  API Documentation
                </a>
                <a href="#" className="block text-sm text-gray-600 hover:text-primary-600 transition-colors py-2">
                  Help & Support
                </a>
              </nav>
              
              <div className="pt-4 border-t border-gray-200">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    <span className="text-sm text-gray-600">AI System Online</span>
                  </div>
                  <button className="p-2 text-gray-400 hover:text-gray-600 transition-colors">
                    <Cog6ToothIcon className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;