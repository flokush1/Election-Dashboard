import React from 'react';
import { formatPreviewValue } from '../../../shared/utils.js';

const DataPreviewTable = ({ title = 'Data Preview', preview, loading, error }) => (
  <div className="bg-white rounded-xl shadow-sm border p-6">
    <h3 className="text-lg font-semibold mb-4">{title}</h3>
    {loading && <p className="text-gray-500">Loading preview...</p>}
    {error && <p className="text-red-500">{error}</p>}
    {preview?.rows?.length > 0 && (
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead>
            <tr>
              {(preview.columns || Object.keys(preview.rows[0])).slice(0, 8).map((col) => (
                <th key={col.name || col} className="text-left p-2 border-b">{col.name || col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.rows.slice(0, 10).map((row, index) => (
              <tr key={index}>
                {(preview.columns || Object.keys(row)).slice(0, 8).map((col) => {
                  const key = col.name || col;
                  return <td key={key} className="p-2 border-b">{formatPreviewValue(row[key], key)}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )}
  </div>
);

export default DataPreviewTable;
