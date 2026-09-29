import test from 'node:test';
import assert from 'node:assert/strict';
import {answerBlocks, renderAnswer} from '../../src/library_ingestor/web/static/answer-format.js';

// Minimal DOM double: verify safe node construction without a browser dependency.
class Node {
  constructor(tag, text = '') { this.tag = tag; this.textContent = text; this.children = []; this.attributes = {}; this.ownerDocument = doc; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener(name, callback) { this[name] = callback; }
}
const doc = {
  createElement: (tag) => new Node(tag),
  createTextNode: (text) => new Node('#text', text),
  createDocumentFragment: () => new Node('#fragment'),
};
const descendants = (node) => [node, ...node.children.flatMap(descendants)];

test('paragraphs, headings, lists and continuation lines retain reading order', () => {
  assert.deepEqual(answerBlocks('Respuesta\ndirecta.\n\n## Símbolos\n- Uno\n  ampliado\n- Dos\n\n1. Primero\n2. Segundo'), [
    {type: 'paragraph', text: 'Respuesta directa.'},
    {type: 'heading', text: 'Símbolos'},
    {type: 'unordered', items: ['Uno ampliado', 'Dos']},
    {type: 'ordered', items: ['Primero', 'Segundo']},
  ]);
});

test('named citations resolve exact excerpts, including within emphasis', () => {
  const target = new Node('div'); let selected;
  renderAnswer(target, '**Idea [1]** y otra [2], desconocida [99].', [
    {number: 1, title: 'Ficciones', page_number: 12},
    {number: 2, title: 'Ficciones', page_number: 40},
  ], (number) => { selected = number; });
  const nodes = descendants(target);
  const citations = nodes.filter((node) => node.tag === 'button');
  assert.equal(citations.length, 2);
  assert.equal(citations[0].textContent, 'Ficciones');
  assert.match(citations[1].title, /p. 40/);
  citations[1].click(); assert.equal(selected, 2);
  assert.ok(nodes.some((node) => node.tag === '#text' && node.textContent === '[99]'));
});

test('HTML and external links remain inert text, including malicious book titles', () => {
  const target = new Node('div');
  renderAnswer(target, '<img src=x onerror=alert(1)> [link](javascript:alert(1)) [1]', [
    {number: 1, title: '<script>alert(1)</script>'},
  ], () => {});
  const nodes = descendants(target);
  assert.ok(nodes.every((node) => ['div', '#fragment', 'p', '#text', 'button'].includes(node.tag)));
  assert.equal(nodes.find((node) => node.tag === 'button').textContent, '<script>alert(1)</script>');
});

test('partial streaming Markdown and empty answers remain renderable', () => {
  const target = new Node('div');
  for (const text of ['', '##', '**Idea', 'Texto [', 'Texto [1', 'Texto [1]']) {
    assert.doesNotThrow(() => renderAnswer(target, text, [], () => {}));
  }
});
