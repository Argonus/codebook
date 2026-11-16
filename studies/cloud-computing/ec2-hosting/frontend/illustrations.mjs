export function catDrawing(id, { enclosure = true } = {}) {
  const fur = { miso: ['#d5a16c', '#e9c297', '#bf8555'], luna: ['#e9dfc5', '#f7edda', '#d3c4a3'], chmurka: ['#97a4a1', '#c4cfca', '#788d8b'] }[id];
  const [body, face, accent] = fur;
  return `<svg class="cat-scene cat-scene--${id}" viewBox="0 0 420 260" aria-hidden="true">
    <defs><linearGradient id="bg-${id}" x2="0" y2="1"><stop stop-color="#e9eddc"/><stop offset="1" stop-color="#f3efdb"/></linearGradient></defs>
    <rect width="420" height="260" fill="url(#bg-${id})"/>
    <circle cx="337" cy="56" r="26" fill="#efe1aa" opacity=".65"/>
    <path d="M0 168Q95 129 190 159T420 155V260H0Z" fill="#d9dfc0"/>
    <path d="M0 204Q125 169 237 198T420 183V260H0Z" fill="#c7d0ac"/>
    <path d="M30 203q-14-35-6-46q15 3 14 32q2-33 14-32q10 15-8 44" fill="#94a77b"/>
    <path d="M362 202q-12-29-4-37q12 5 10 25q4-31 15-27q7 12-14 40" fill="#93a47a"/>
    <ellipse cx="301" cy="218" rx="63" ry="14" fill="#b4bd99" opacity=".5"/>
    <path d="M248 214q4-17 53-17q48 0 54 17q-6 14-54 13q-48 1-53-13Z" fill="#c69378" stroke="#a87c62" stroke-width="2"/>
    <path d="M258 213q36 10 86 0" stroke="#e1b397" stroke-width="2" fill="none"/>
    ${enclosure ? `<g class="enclosure-back" stroke="#927d5a" stroke-width="3" stroke-linejoin="round">
      <path d="M69 106 94 80h113l24 26v111H69Z" fill="#ded0ab"/>
      <path d="M83 108h133v99H83Z" fill="#b8b598"/>
      <path d="M77 105h146M77 211h146" stroke-width="7"/>
      <path d="M81 109v99m137-99v99" stroke-width="6"/>
    </g>` : ''}
    <g class="cat-position"><g class="living-cat" stroke="#625d4c" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
      <ellipse cx="149" cy="210" rx="48" ry="7" fill="#6b7657" opacity=".16" stroke="none"/>
      <path class="cat-tail" d="M122 193c-23 7-36-1-34-15c1-9 11-13 15-4" fill="none" stroke="${accent}" stroke-width="12"/>
      <path d="M120 179q-2-35 29-36q34-1 31 38l-6 27h-52Z" fill="${body}"/>
      <ellipse cx="151" cy="184" rx="19" ry="23" fill="${face}" stroke="none"/>
      <path class="cat-ear cat-ear-left" d="m120 144-5-34 24 16" fill="${body}"/>
      <path class="cat-ear cat-ear-right" d="m166 126 22-16-3 35" fill="${body}"/>
      <path d="m121 132-2-13 12 10m45 0 9-10-1 15" fill="#d49f93" stroke="none"/>
      <path d="M115 146q-1-23 37-24q38 0 37 24q-1 32-37 31q-36 0-37-31Z" fill="${face}"/>
      ${id === 'miso' ? `<path d="m141 125 3 12m9-13v12m10-11-3 12m-43 11 10 2m49-1 10-3" stroke="${accent}" stroke-width="4"/>` : ''}
      ${id === 'chmurka' ? `<path d="M127 131q12-9 25-7q23-4 30 13l-15 9-14-7-14 7Z" fill="${body}" stroke="none"/>` : ''}
      <g class="cat-eyes"><path d="M132 149q4-4 8 0m23 0q4-4 8 0" fill="none"/></g>
      <path d="m148 155 4 3 4-3" fill="#bd8b83" stroke="#9b766c"/><path d="M152 159v3m0 0q-5 5-9 0m9 0q5 5 9 0" fill="none" stroke-width="1.5"/>
      <path d="m121 155-15-3m17 9-15 2m72-8 15-3m-16 9 15 2" stroke="#857b67" stroke-width="1.3"/>
      <path d="M132 191v17h12v-8m15 0v8h13v-17" fill="${body}"/>
      <path d="M142 178q11 4 21 0" stroke="${accent}" stroke-width="4"/>
      <circle cx="153" cy="182" r="3" fill="#ceb568" stroke-width="1"/>
    </g></g>
    ${enclosure ? `<g class="enclosure-door" stroke="#8c795b" stroke-linecap="round">
      <path d="M84 111h132v96H84Z" fill="#f5f0d7" fill-opacity=".06" stroke-width="3"/>
      <path d="M106 113v91m22-91v91m22-91v91m22-91v91m22-91v91" stroke-width="2.6" opacity=".72"/>
      <path d="M201 150v14" stroke-width="5"/>
    </g>` : ''}
    <path d="M52 216h25l-4 9H56Z" fill="#9aab97" stroke="#77856c" stroke-width="2"/>
    <g fill="#ece3b8"><circle cx="45" cy="237" r="2.5"/><circle cx="384" cy="221" r="2.5"/><circle cx="231" cy="234" r="2"/></g>
    <g stroke="#96a47f" stroke-width="2" stroke-linecap="round"><path d="m19 234 3-9 4 7m369 4 3-10 4 8"/></g>
  </svg>`;
}
