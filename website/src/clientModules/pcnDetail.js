// Fills the detail panel under a verified chat answer on the training notebook
// pages. In Jupyter an inline script does this; the site removes inline scripts
// from notebook outputs, so the same behaviour is attached here after each route.

function wire() {
  document.querySelectorAll('[id^="pcn-chat-"]').forEach((card) => {
    if (card.dataset.pcnWired) return;
    card.dataset.pcnWired = '1';
    const panel = card.querySelector('.pcn-detail');
    if (!panel) return;
    card.querySelectorAll('.pcn-mark').forEach((mark) => {
      const tip = mark.querySelector('.pcn-tip');
      if (!tip) return;
      const show = () => {
        panel.innerHTML = tip.innerHTML;
        panel.style.color = '#16315a';
      };
      mark.addEventListener('mouseenter', show);
      mark.addEventListener('focus', show);
    });
  });
}

export function onRouteDidUpdate() {
  if (typeof document !== 'undefined') {
    window.requestAnimationFrame(wire);
  }
}
