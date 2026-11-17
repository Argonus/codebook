export function dogDrawing(id, { enclosure = true } = {}) {
  const [fur, muzzle, ear] = {
    miso: ['#c99662', '#f3dfc5', '#9e6e46'],
    luna: ['#e6d5b8', '#fcf2df', '#c2a078'],
    chmurka: ['#8f9fae', '#e3ebef', '#647a8e'],
  }[id];
  return `<svg class="dog-scene dog-scene--${id}" viewBox="0 0 420 260" aria-hidden="true">
    <defs><linearGradient id="bg-${id}" x2="0" y2="1"><stop stop-color="#e8f4ff"/><stop offset="1" stop-color="#d4e7f6"/></linearGradient></defs>
    <rect width="420" height="260" fill="url(#bg-${id})"/>
    <circle cx="337" cy="56" r="26" fill="#fff1c7" opacity=".8"/>
    <path d="M34 66q5-15 19-8q12-17 25-3q16-3 20 11Z" fill="#fff" opacity=".65"/>
    <path d="M0 168Q95 129 190 159T420 155V260H0Z" fill="#c2dbe9"/>
    <path d="M0 204Q125 169 237 198T420 183V260H0Z" fill="#abc8dc"/>
    <path d="M30 203q-14-35-6-46q15 3 14 32q2-33 14-32q10 15-8 44M362 202q-12-29-4-37q12 5 10 25q4-31 15-27q7 12-14 40" fill="#82a8c2"/>
    <ellipse cx="301" cy="218" rx="63" ry="14" fill="#789eb8" opacity=".3"/>
    <path d="M248 214q4-17 53-17q48 0 54 17q-6 14-54 13q-48 1-53-13Z" fill="#6caddb" stroke="#4f8ab7" stroke-width="2"/>
    <path d="M258 213q36 10 86 0" stroke="#b3d7ef" stroke-width="2" fill="none"/>
    ${enclosure ? `<g class="enclosure-back" stroke="#7c98ac" stroke-width="3" stroke-linejoin="round">
      <path d="M69 106 94 80h113l24 26v111H69Z" fill="#d9e4eb"/>
      <path d="M83 108h133v99H83Z" fill="#a4bbc9"/>
      <path d="M77 105h146M77 211h146" stroke-width="7"/>
      <path d="M81 109v99m137-99v99" stroke-width="6"/>
    </g>` : ''}
    <g class="dog-position"><g class="living-dog" stroke="#536576" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
      <ellipse cx="149" cy="210" rx="48" ry="7" fill="#527c98" opacity=".18" stroke="none"/>
      <path class="dog-tail" d="M124 189c-25-2-34-15-28-31" fill="none" stroke="${ear}" stroke-width="12"/>
      <path d="M121 179q-2-34 29-36q34-1 31 38l-6 27h-53Z" fill="${fur}"/>
      <ellipse cx="151" cy="185" rx="18" ry="21" fill="${muzzle}" stroke="none"/>
      <path class="dog-ear-left" d="M130 119q-19-13-28 7q-10 24 1 42q8 11 17-6l13-30Z" fill="${ear}"/>
      <path class="dog-ear-right" d="M172 119q19-13 28 7q10 24-1 42q-8 11-17-6l-13-30Z" fill="${ear}"/>
      <path d="M116 140q0-27 35-26q34-1 35 26q2 31-35 33q-37-2-35-33Z" fill="${fur}"/>
      ${id === 'miso' ? `<ellipse cx="132" cy="137" rx="12" ry="16" fill="${ear}" stroke="none"/>` : ''}
      ${id === 'chmurka' ? `<path d="m149 116-9 30h23l-10-30Z" fill="${muzzle}" stroke="none"/>` : ''}
      <ellipse cx="151" cy="158" rx="21" ry="14" fill="${muzzle}" stroke="none"/>
      <g fill="#344959" stroke="none"><ellipse cx="133" cy="141" rx="3" ry="4"/><ellipse cx="169" cy="141" rx="3" ry="4"/></g>
      <g fill="#fff" stroke="none"><circle cx="134" cy="140" r="1"/><circle cx="170" cy="140" r="1"/></g>
      <path d="M144 151q7-4 14 0q0 7-7 8q-7-1-7-8Z" fill="#405565"/>
      <path d="M151 159v4m0 0q-6 6-12 0m12 0q6 6 12 0" fill="none" stroke-width="1.6"/>
      <path d="M146 168v5q5 7 10 0v-5" fill="#de9ca1" stroke="#b57d89" stroke-width="1.2"/>
      <path d="M130 192v16h14v-9m15 0v9h15v-16" fill="${fur}"/>
      <path d="M126 176q25 10 50 0" stroke="#367fba" stroke-width="5" fill="none"/>
      <path d="M148 181h7l2 7-5 4-5-4Z" fill="#f0c574" stroke="#b68e4b" stroke-width="1.2"/>
    </g></g>
    ${enclosure ? `<g class="enclosure-door" stroke="#7190a7" stroke-linecap="round">
      <path d="M84 111h132v96H84Z" fill="#eef7ff" fill-opacity=".06" stroke-width="3"/>
      <path d="M106 113v91m22-91v91m22-91v91m22-91v91m22-91v91" stroke-width="2.6" opacity=".72"/>
      <path d="M201 150v14" stroke-width="5"/>
    </g>` : ''}
    <path d="M52 216h25l-4 9H56Z" fill="#88b5d5" stroke="#648ba8" stroke-width="2"/>
    <g fill="#edf7ff"><circle cx="45" cy="237" r="2.5"/><circle cx="384" cy="221" r="2.5"/><circle cx="231" cy="234" r="2"/></g>
  </svg>`;
}
