import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, message: '' };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, message: error.message };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-red-50 p-6">
          <div className="bg-white rounded-lg shadow p-8 max-w-lg text-center">
            <h2 className="text-2xl font-bold text-red-600 mb-3">Something went wrong</h2>
            <p className="text-gray-600 mb-4">{this.state.message || 'Please reload the dashboard.'}</p>
            <button className="bg-red-600 text-white px-4 py-2 rounded" onClick={() => window.location.reload()}>
              Reload
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;
