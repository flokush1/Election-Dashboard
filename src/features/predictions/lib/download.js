export const downloadRows = (rows, columns, filename, type = 'csv') => {
  if (!rows?.length) return;
  const cols = columns?.length ? columns : Array.from(new Set(rows.flatMap((row) => Object.keys(row))));
  let blob;
  if (type === 'json') {
    blob = new Blob([JSON.stringify(rows, null, 2)], { type: 'application/json' });
  } else {
    const esc = (val) => {
      if (val === null || val === undefined) return '';
      const s = String(val).replace(/"/g, '""');
      return /[",\n]/.test(s) ? `"${s}"` : s;
    };
    const lines = [cols.join(',')];
    for (const row of rows) lines.push(cols.map((col) => esc(row[col])).join(','));
    blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};
