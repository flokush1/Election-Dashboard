export const allocateVotes = (ratios, total) => {
  const parties = Object.keys(ratios || {});
  const exact = parties.map((p) => ({ p, exact: (ratios[p] || 0) * total }));
  const floorVotes = {};
  let sumFloors = 0;
  exact.forEach((e) => {
    const f = Math.floor(e.exact);
    floorVotes[e.p] = f;
    sumFloors += f;
  });
  let remaining = total - sumFloors;
  exact.sort((a, b) => (b.exact - Math.floor(b.exact)) - (a.exact - Math.floor(a.exact)));
  let i = 0;
  while (remaining > 0 && i < exact.length) {
    floorVotes[exact[i].p] += 1;
    remaining -= 1;
    i += 1;
  }
  return floorVotes;
};

export const allocateBoothVotes = (ratios, totalPolled = 0) => {
  const partyVotes = allocateVotes(ratios, totalPolled);
  const totalAllocated = Object.values(partyVotes).reduce((sum, value) => sum + value, 0);
  if (totalPolled !== totalAllocated) {
    const diff = totalPolled - totalAllocated;
    if (diff !== 0) {
      const maxParty = Object.entries(partyVotes).sort((a, b) => b[1] - a[1])[0]?.[0];
      if (maxParty) partyVotes[maxParty] += diff;
    }
  }
  const orderedAllocated = Object.entries(partyVotes).sort((a, b) => b[1] - a[1]);
  return {
    partyVotes,
    orderedAllocated,
    winnerParty: orderedAllocated[0]?.[0] || 'Unknown',
    recomputedMargin: orderedAllocated.length >= 2
      ? orderedAllocated[0][1] - orderedAllocated[1][1]
      : 0
  };
};
