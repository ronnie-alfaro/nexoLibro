// Deliberately small Markdown subset. Model output always becomes text nodes,
// never HTML, URLs, styles, or executable markup.
export function answerBlocks(text) {
  const blocks = [];
  let paragraph = [];
  let list = null;
  const flush = () => {
    if (paragraph.length) blocks.push({type: 'paragraph', text: paragraph.join(' ')});
    paragraph = [];
    list = null;
  };
  for (const line of text.replace(/\r\n?/g, '\n').split('\n')) {
    if (!line.trim()) { flush(); continue; }
    const heading = line.match(/^#{1,6}\s+(.+?)\s*#*$/);
    const item = line.match(/^\s*(?:([-*+])|\d+[.)])\s+(.+)$/);
    if (heading) {
      flush(); blocks.push({type: 'heading', text: heading[1]});
    } else if (item) {
      const type = item[1] ? 'unordered' : 'ordered';
      if (!list || list.type !== type) {
        flush(); list = {type, items: []}; blocks.push(list);
      }
      list.items.push(item[2]);
    } else if (list && /^\s{2,}\S/.test(line)) {
      list.items[list.items.length - 1] += ' ' + line.trim();
    } else {
      list = null; paragraph.push(line.trim());
    }
  }
  flush();
  return blocks;
}

export function inlineParts(text) {
  const parts = [];
  let cursor = 0;
  for (const match of text.matchAll(/\[(\d+)\]|\*\*([^*\n]+)\*\*|\*([^*\n]+)\*/g)) {
    if (match.index > cursor) parts.push({type: 'text', text: text.slice(cursor, match.index)});
    if (match[1]) parts.push({type: 'citation', number: Number(match[1]), text: match[0]});
    else parts.push({type: match[2] ? 'strong' : 'em', text: match[2] || match[3]});
    cursor = match.index + match[0].length;
  }
  if (cursor < text.length) parts.push({type: 'text', text: text.slice(cursor)});
  return parts;
}

export function renderAnswer(target, text, sources, onCitation) {
  const doc = target.ownerDocument;
  const byNumber = new Map(sources.map((source) => [source.number, source]));
  function inline(parent, value, emphasis = true) {
    for (const part of inlineParts(value)) {
      const source = part.type === 'citation' && byNumber.get(part.number);
      if (source) {
        const button = doc.createElement('button');
        button.type = 'button'; button.className = 'citation';
        button.textContent = source.title || source.filename;
        const location = [source.author, source.section, source.page_number ? `p. ${source.page_number}` : null].filter(Boolean).join(' · ');
        button.title = `${source.title || source.filename}${location ? ' — ' + location : ''} [${part.number}]`;
        button.setAttribute('aria-label', `Ver fuente ${part.number}: ${button.title}`);
        button.addEventListener('click', () => onCitation(part.number));
        parent.append(button);
      } else if (emphasis && (part.type === 'strong' || part.type === 'em')) {
        const node = doc.createElement(part.type);
        inline(node, part.text, false); parent.append(node);
      } else parent.append(doc.createTextNode(part.text));
    }
  }
  const fragment = doc.createDocumentFragment();
  for (const block of answerBlocks(text)) {
    const tag = {heading: 'h3', paragraph: 'p', unordered: 'ul', ordered: 'ol'}[block.type];
    const node = doc.createElement(tag);
    if (block.items) {
      for (const item of block.items) {
        const li = doc.createElement('li'); inline(li, item); node.append(li);
      }
    } else inline(node, block.text);
    fragment.append(node);
  }
  target.replaceChildren(fragment);
}
