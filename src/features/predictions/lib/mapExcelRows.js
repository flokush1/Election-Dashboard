export function mapExcelRows(objectRows, orderedColumns = []) {
  return objectRows.map((row, index) => {
    const findValue = (names, fallback = '') => {
      for (const name of names) {
        const match = Object.keys(row).find((key) => key.toLowerCase().trim() === name.toLowerCase().trim());
        if (match && row[match] !== undefined && row[match] !== null && String(row[match]).trim() !== '') {
          return row[match];
        }
      }
      return fallback;
    };
    return {
      voter_id: String(findValue(['voters id', 'voter id', 'voter_id', 'epic'], `VOTER_${String(index + 1).padStart(5, '0')}`)),
      name: findValue(['name', 'voter_name'], 'Unknown'),
      age: findValue(['age'], 30),
      gender: String(findValue(['gender', 'sex'], 'Unknown')).toUpperCase(),
      religion: String(findValue(['religion'], 'HINDU')).toUpperCase(),
      caste: String(findValue(['caste', 'category'], '')).toUpperCase(),
      economic_category: String(findValue(['economic_category', 'economic'], 'MIDDLE CLASS')).toUpperCase(),
      locality: String(findValue(['Locality', 'locality'], '')).toUpperCase(),
      assembly: findValue(['assembly name', 'assembly'], 'Unknown'),
      columns: orderedColumns
    };
  });
}
