// Map UTF-8 token ranges to intact graphemes and UTF-16 textarea selections.
export function mapText(text, byteLists) {
  let offset = 0;
  const tokens = byteLists.map((bytes) => {
    const start = offset;
    offset += bytes.length;
    return { start, end: offset };
  });
  const segments = [];
  let byteOffset = 0,
    firstToken = 0;
  for (const { segment, index } of new Intl.Segmenter(undefined, {
    granularity: "grapheme",
  }).segment(text)) {
    const end = byteOffset + new TextEncoder().encode(segment).length;
    const indices = [];
    while (firstToken < tokens.length && tokens[firstToken].end <= byteOffset)
      firstToken++;
    for (let i = firstToken; i < tokens.length && tokens[i].start < end; i++)
      indices.push(i);
    segments.push({
      text: segment,
      start: index,
      end: index + segment.length,
      byteStart: byteOffset,
      byteEnd: end,
      indices,
    });
    byteOffset = end;
  }
  return { tokens, segments };
}
