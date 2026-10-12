// Which cookbooks are public. The list lives in website/cookbook-visibility.json
// so that the site config, the content modules, and scripts/docs/check_cookbooks.py
// read the same file.
import visibility from '../../cookbook-visibility.json';

export const hiddenCookbooks = visibility.hidden;

export function isHiddenCookbook(id) {
  return hiddenCookbooks.includes(id);
}

// True for a site path into a hidden cookbook, for example
// "/cookbook/governance-and-risk/position".
export function isHiddenPath(to) {
  const m = /^\/cookbook\/([^/#?]+)/.exec(to || '');
  return Boolean(m) && hiddenCookbooks.includes(m[1]);
}

const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty'];

// "Eleven" for 11, so that counts in running text follow the visible list.
export function countWord(n, capitalize = false) {
  const w = WORDS[n] || String(n);
  return capitalize ? w.charAt(0).toUpperCase() + w.slice(1) : w;
}
