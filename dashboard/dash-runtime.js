/*
 * Minimal runtime that lets page.html run on any static host. It implements only
 * what the page uses:
 *   dash.data(id)        -> {status, data, columns, message}   (CSV loaded with d3.csv)
 *   dash.draw(name, fn)  -> run fn(el) on every visible [data-draw=name]
 *   dash.onData(fn), dash.setLink(name), dash.colors, dash.params() & friends (no-ops)
 * Drawings re-run when data arrives, when the window width changes and when a
 * `hidden` attribute is removed (tabs).
 */
(function () {
  const SOURCES = window.DASH_SOURCES || {};
  const store = {};
  const draws = new Map();
  const onDataFns = [];
  let pending = false;

  function load(id) {
    if (store[id]) return;
    store[id] = { status: 'loading', data: [], columns: [] };
    d3.csv(SOURCES[id])
      .then((rows) => { store[id] = { status: 'ok', data: rows, columns: rows.columns || [] }; schedule(); })
      .catch((e) => { store[id] = { status: 'error', data: [], columns: [], message: String(e) }; console.error('dash: failed to load', id, e); schedule(); });
  }

  function runDraw(name) {
    const fn = draws.get(name);
    if (!fn) return;
    document.querySelectorAll(`[data-draw="${CSS.escape(name)}"]`).forEach((el) => {
      if (el.closest('[hidden]')) return;
      try { fn(el); } catch (e) { console.error(`dash: drawing "${name}" failed`, e); }
    });
  }

  function runAll() {
    pending = false;
    onDataFns.forEach((fn) => { try { fn(); } catch (e) { console.error(e); } });
    draws.forEach((_, name) => runDraw(name));
  }

  function schedule() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(runAll);
  }

  window.dash = {
    data(id) {
      if (!(id in SOURCES)) return { status: 'missing', data: [], columns: [] };
      load(id);
      return store[id];
    },
    draw(name, fn) {
      if (typeof fn === 'function') draws.set(name, fn);
      runDraw(name);
    },
    onData(fn) { onDataFns.push(fn); try { fn(); } catch (e) { console.error(e); } },
    setLink(name) { try { history.replaceState(null, '', '#' + name); } catch (e) { /* ignore */ } },
    colors: ['#3a86c8', '#e0603f', '#4f9e4a', '#9a6fe0', '#d4a20a', '#2aa198', '#c2477a', '#7a7f8c'],
    params() { return {}; }, setParams() {}, resetParams() {},
    calc() {}, loader() {}, refresh() {},
  };

  // Re-run drawings when a hidden section is shown, and when the width changes. Tooltips
  // are skipped: redrawing a chart removes its tooltip, so one could never appear.
  new MutationObserver((muts) => {
    if (muts.some((m) => m.attributeName === 'hidden' && !m.target.hidden && !m.target.matches('.ml-tip'))) schedule();
  }).observe(document.documentElement, { attributes: true, attributeFilter: ['hidden'], subtree: true });
  let lastW = window.innerWidth;
  window.addEventListener('resize', () => { if (window.innerWidth !== lastW) { lastW = window.innerWidth; schedule(); } });
})();
