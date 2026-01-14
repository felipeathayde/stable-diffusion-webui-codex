// ==UserScript==
// @name         Gemini: auto-download imagens (googleusercontent/gg)
// @namespace    local
// @version      0.2.1
// @description  Monkeypatch fetch/XHR e baixa automaticamente imagens https://lh*.googleusercontent.com/gg/... quando aparecerem no Network (ou dentro do StreamGenerate), sem injetar <script> (Trusted Types safe).
// @match        https://gemini.google.com/*
// @run-at       document-start
// @grant        GM_download
// @connect      lh3.googleusercontent.com
// @connect      *.googleusercontent.com
// ==/UserScript==


(function () {
  'use strict';

  const DEBUG = true;
  const TRACE_MATCHES = true; // logar contagens e exemplos de match no StreamGenerate
  const SAVE_AS = false; // true = perguntar onde salvar
  const TARGET_RE = /^https?:\/\/(?:[a-z0-9-]+\.)?googleusercontent\.com\/gg\//i;

  const w = typeof unsafeWindow !== 'undefined' ? unsafeWindow : window;
  const pageConsole =
    w && w.console && typeof w.console.log === 'function' ? w.console : console;
  
  if (w.__tm_gemini_imgdl_installed) return;
  w.__tm_gemini_imgdl_installed = true;

  const state =
    w.__tm_gemini_imgdl_state ||
    (w.__tm_gemini_imgdl_state = {
      installedAt: new Date().toISOString(),
      fetchSeen: 0,
      xhrSeen: 0,
      streamBodiesScanned: 0,
      urlMatches: 0,
      pairMatches: 0,
      downloadsScheduled: 0,
      lastStreamUrl: null,
      lastStreamBodyLen: null,
      lastMatchesSample: [],
      fetchUrls: [],
      xhrUrls: [],
      scanAttempts: 0,
      scanSkips: 0,
      scanErrors: 0,
      scanEvents: [],
      bodyHasGoogleusercontent: 0,
      bodyHasLhAny: 0,
    
    // Metrics
    metrics: {
        takedowns: 0,
        lastRefusal: null,
        safetyScores: []
    }
  });
    
  // -- Aggressive Patch Maintenance helpers --
  let myFetch = w.fetch;
  let myXHR = w.XMLHttpRequest;

  w.__tm_gemini_imgdl_dump = () => {
    pageConsole.log('[TM Gemini DL] state', state);
    return state;
  };

  function pushRing(arr, item, maxLen) {
    try {
      if (!Array.isArray(arr)) return;
      arr.push(item);
      while (arr.length > maxLen) arr.shift();
    } catch (_) {}
  }

  // Back-compat init
  if (!Array.isArray(state.fetchUrls)) state.fetchUrls = [];
  if (!Array.isArray(state.xhrUrls)) state.xhrUrls = [];
  if (!Array.isArray(state.scanEvents)) state.scanEvents = [];
  if (!state.metrics) state.metrics = { takedowns: 0, lastRefusal: null, safetyScores: [] };
  if (!state.candidates) state.candidates = []; // Initialize candidates if not present

  const downloaded =
    w.__tm_gemini_imgdl_downloaded && typeof w.__tm_gemini_imgdl_downloaded.has === 'function'
      ? w.__tm_gemini_imgdl_downloaded
      : (w.__tm_gemini_imgdl_downloaded = new Set());

  function log(...args) {
    if (DEBUG) pageConsole.log('[TM Gemini DL]', ...args);
  }
  
  // Overlay UI
  let overlayEl = null;

  function updateOverlay() {
      if (!overlayEl) return;
      // Get short location
      let shortLoc = '...';
      try {
          const href = w.location.href;
          shortLoc = href.slice(-40);
      } catch(e) { }
      
      const lines = [];
      lines.push('TM Gemini: ON');
      lines.push(`Ctx: ${shortLoc}`);
      lines.push(`Fetch: ${state.fetchSeen} | XHR: ${state.xhrSeen}`);
      lines.push(`Scan: ${state.scanAttempts} (err: ${state.scanErrors})`);
      lines.push('Candidates: ' + state.candidates.length);
      lines.push('Matches: ' + (state.pairMatches + state.urlMatches));
      lines.push('DL: ' + state.downloadsScheduled);
    
      if (state.metrics.takedowns > 0) {
          lines.push('--- SAFETY ---');
          lines.push('Takedowns: ' + state.metrics.takedowns);
          if (state.metrics.lastRefusal) lines.push('Refusal: ' + state.metrics.lastRefusal.slice(0, 20) + '...');
      }

      if (state.metrics.safetyScores.length > 0) {
          lines.push('--- CLASSIFIERS ---');
          state.metrics.safetyScores.forEach(item => {
               lines.push(`${item.label}: ${item.score.toFixed(4)}`);
          });
      }

      if (state.lastMatchesSample.length) {
          lines.push('--- SAMPLE ---');
          state.lastMatchesSample.forEach(s => {
              let n = s.name || '(no-name)';
              if (n.length > 20) n = '...' + n.slice(-17);
              lines.push(n);
          });
      }
      overlayEl.textContent = lines.join('\n');
  }
  
  function createOverlay() {
      if (document.getElementById('tm-gemini-overlay')) return;
      const container = document.createElement('div');
      container.id = 'tm-gemini-overlay';
      container.style.position = 'fixed';
      container.style.top = '10px';
      container.style.right = '10px';
      container.style.zIndex = '999999';
      container.style.border = '1px solid #0f0';
      container.style.backgroundColor = 'black';
      container.style.color = '#0f0';
      container.style.padding = '4px';
      container.style.opacity = '0.9';
      container.style.fontFamily = 'monospace';
      container.style.fontSize = '12px';
      container.style.lineHeight = '1.2';
      
      // Stats display
      const statsDiv = document.createElement('div');
      statsDiv.style.whiteSpace = 'pre';
      statsDiv.style.pointerEvents = 'none';
      container.appendChild(statsDiv);
      overlayEl = statsDiv;
      
      // Manual Re-Patch button
      const btn = document.createElement('button');
      btn.textContent = '🔄 Re-Patch';
      btn.style.marginTop = '4px';
      btn.style.display = 'block';
      btn.style.cursor = 'pointer';
      btn.style.backgroundColor = '#333';
      btn.style.color = '#0f0';
      btn.style.border = '1px solid #0f0';
      btn.style.padding = '2px 6px';
      btn.style.fontFamily = 'monospace';
      btn.style.fontSize = '11px';
      btn.onclick = function() {
          log('Manual Re-Patch triggered by user');
          doPatchFetch();
          doPatchXHR();
          updateOverlay();
      };
      container.appendChild(btn);
      
      document.body.appendChild(container);
      updateOverlay();
  }
  
  if (document.body) {
      createOverlay();
  } else {
      document.addEventListener('DOMContentLoaded', createOverlay);
  }

  function fnv1a32(str) {
    let h = 0x811c9dc5;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) >>> 0;
    }
    return ('0000000' + h.toString(16)).slice(-8);
  }

  function safeFilename(name) {
    return String(name).replace(/[\\/:*?"<>|]+/g, '_').slice(0, 180);
  }

  function defaultNameForUrl(url) {
    return `gemini-${fnv1a32(url)}.jpg`;
  }

  function scheduleDownload(url, nameHint) {
    if (!url) return;
    if (downloaded.has(url)) {
        log('duplicate skipped', url.slice(-20));
        return;
    }
    
    downloaded.add(url);
    state.downloadsScheduled += 1;
    updateOverlay();

    const name = safeFilename(nameHint || defaultNameForUrl(url));
    log('ATTEMPTING DOWNLOAD', name, url);

    if (typeof GM_download !== 'function') {
        console.error('[TM Gemini DL] GM_download MISSING! Check script headers/Tampermonkey settings.');
        alert('GM_download missing!');
        return;
    }

    GM_download({
      url,
      name,
      saveAs: SAVE_AS,
      onload: () => log('GM_download success', name),
      onerror: (e) => console.error('[TM Gemini DL] GM_download FAILED', name, e),
    });
  }

  function normalizeUrl(raw) {
    if (!raw) return '';
    let u = String(raw);
    // Replace unicode escapes
    u = u.replace(/\\u002F/gi, '/');
    u = u.replace(/\\u003A/gi, ':');
    // Replace escaped slashes \/ or \\/
    u = u.replace(/\\+\//g, '/'); 
    
    if (u.indexOf('//') === 0) {
        u = 'https:' + u;
    }
    
    // Trim garbage
    const trimChars = ['\\', ']', ')', '"', "'", ',', ' ', '\t', '\r', '\n'];
    while (u.length > 0 && trimChars.indexOf(u[u.length - 1]) !== -1) {
        u = u.slice(0, -1);
    }
    return u;
  }

  function isTarget(url) {
    return TARGET_RE.test(normalizeUrl(url));
  }

  function shouldScanUrl(url) {
    const u = String(url || '');
    return /BardChatUi\/data\//i.test(u) || /StreamGenerate|batchexecute/i.test(u);
  }

  function extractFromText(text, opts) {
    const isChunk = !!(opts && opts.chunk);
    if (!text || typeof text !== 'string') return;
    if (text.length > 5 * 1024 * 1024) text = text.slice(0, 5 * 1024 * 1024);
    if (!isChunk) state.streamBodiesScanned += 1;

    // DEBUG: Force log candidates to debug Regex failure
    const hasGGC = /googleusercontent/i.test(text);
    if (hasGGC) state.bodyHasGoogleusercontent += 1;
    if (/lh\d+\.googleusercontent/i.test(text)) state.bodyHasLhAny += 1;
    
    if (DEBUG && hasGGC) {
         console.log('[TM Gemini DL] FOUND CANDIDATE (isChunk=' + isChunk + ')');
         // Find the position and log context
         const matchIdx = text.search(/googleusercontent/i);
         if (matchIdx !== -1) {
             const start = Math.max(0, matchIdx - 200);
             const end = Math.min(text.length, matchIdx + 400);
             console.log('[TM Gemini DL] CANDIDATE CONTEXT:', text.slice(start, end));
         }
    }

    // MANUAL PARSING: Simply look for googleusercontent... and grabs strictly between " " or \" \"
    
    let pairMatches = 0;
    let urlMatches = 0;
    const sample = [];
    
    let cursor = 0;
    const target = 'googleusercontent.com/gg/';
    
    while (true) {
        const idx = text.indexOf(target, cursor);
        if (idx === -1) break;
        cursor = idx + target.length; 
        
        // Find END of URL (first quote or whitespace)
        // scan forward from idx
        let end = -1;
        for (let i = idx; i < text.length; i++) {
             // If we hit " or \", stop.
             if (text[i] === '"') { end = i; break; }
             if (text[i] === '\\' && text[i+1] === '"') { end = i; break; } // hit escaped quote
        }
        if (end === -1) { cursor++; continue; } // invalid
        
        // Find START of URL (first quote looking backwards)
        let start = -1;
        for (let i = idx - 1; i >= Math.max(0, idx - 300); i--) {
             if (text[i] === '"') { start = i + 1; break; }
             if (text[i] === '"' && text[i-1] === '\\') { start = i + 1; break; } // preceded by escape? NO.
             // If we see \"  at i-1, i. 
             // text[i] is "
             // text[i-1] is \
             if (text[i] === '"') {
                 // Check if escaped
                 if (text[i-1] === '\\') { // Found \"
                     start = i + 1; 
                     break;
                 } else { // Found "
                     start = i + 1;
                     break;
                 }
             }
        }
        if (start === -1) { cursor++; continue; }
        
        const rawUrl = text.slice(start, end);
        const url = normalizeUrl(rawUrl);
        
        // --- Look for Name ---
        // Look at text BEFORE start quote.
        // Expecting: "Name.jpg",
        // Or: \"Name.jpg\",
        
        let name = undefined;
        const pre = text.slice(Math.max(0, start - 300), start); // chunk before URL start
        
        // We want the last string in quotes before the comma implies the URL connection
        // Regex for "Name.jpg" followed by anything until end of string
        // The payload usually looks like: ... \"Name.jpg\" , \"https://...
        // So we look for: \"(something.jpg)\" [\s,]* $
        
        // Match: "([^"]+.(jpg|png|webp))" ... separator ... $
        // Payload: \"Name.jpg\",\"https
        // We are matching against "Name.jpg\",\"
        
        // Relaxed regex: Look for extension .jpg/.png followed by quote/escape
        const nameRx = /([a-zA-Z0-9 _-]+\.(?:jpg|jpeg|png|webp))/i;
        const m = nameRx.exec(pre);
        if (m) {
            name = m[1];
            pairMatches += 1;
        } else {
            urlMatches += 1;
        }

        if (DEBUG) console.log('[TM Gemini DL] Manual Parser Found:', { name, url });
        
        scheduleDownload(url, name);
        if (sample.length < 5) sample.push({ name, url });
    }

    // Also match escaped quotes for standalone URLs
    const urlRe =
      /(?:https?:)?[\/\\]{2,}(?:[a-z0-9-]+\.)?googleusercontent\.com[\/\\]+gg[\/\\]+[^\s"'\\)]+/g;

    for (let m; (m = urlRe.exec(text)); ) {
      urlMatches += 1;
      const url = normalizeUrl(m[0]);
      scheduleDownload(url, undefined);
      if (sample.length < 5) sample.push({ name: undefined, url });
    }

    state.pairMatches += pairMatches;
    state.urlMatches += urlMatches;
    state.lastStreamBodyLen = text.length;
    state.lastMatchesSample = sample;
    updateOverlay();

    if (TRACE_MATCHES && (pairMatches || urlMatches)) {
      log('matches', { pairMatches, urlMatches, sample });
    }
    
    // --- Classifier Extraction ---
    try {
        // Look for the "grail_t2i_image_output_classifier" block or similar
        // The structure usually looks like: ["grail_t2i_image_output_classifier",[null,...,[[["key",score],...]]]]]
        const classifierRegex = /\["(?:grail_t2i_image_output_classifier|image_and_video_safety_classifier|text_safety_classifier)"\s*,\s*\[(?:null\s*,\s*)*\[\[((?:\["[^"]+"\s*,\s*[\d.E-]+\],?)+)\]\]/g;
        
        let cm;
        while ((cm = classifierRegex.exec(text)) !== null) {
            const rawInner = cm[1]; // The part with ["key", score], ["key", score]...
            // Manually parse inner items to avoid JSON parse errors on partial strings
            const itemRegex = /\["([^"]+)"\s*,\s*([\d.E-]+)\]/g;
            let im;
            while ((im = itemRegex.exec(rawInner)) !== null) {
                const key = im[1];
                const score = parseFloat(im[2]);
                
                // Filter for interesting keys to avoid clutter
                if (key.includes('image_output') || key.includes('csam') || key.includes('porn') || key.includes('pedo')) {
                    // Update state.metrics.safetyScores
                    // Check if already exists to update or push new
                    const existing = state.metrics.safetyScores.find(s => s.label === key);
                    if (existing) {
                        existing.score = score;
                    } else {
                        state.metrics.safetyScores.push({ label: key, score: score });
                    }
                    
                    // Keep only top 5 highest scores to prevent overlay overflow
                    state.metrics.safetyScores.sort((a, b) => b.score - a.score);
                    if (state.metrics.safetyScores.length > 6) {
                        state.metrics.safetyScores.length = 6;
                    }
                    
                    if (score > 0.5) {
                        console.warn(`[TM Gemini DL] High Safety Score: ${key} = ${score}`);
                    }
                }
            }
        }
    } catch (e) {
        console.error('[TM Gemini DL] Error parsing classifiers', e);
    }
    // ----------------------------
    
    // Metrics: Look for refusal signals
    if (text.includes('image_output_foundational3p0_takedown')) {
        state.metrics.takedowns += 1;
        state.metrics.lastRefusal = 'Foundational Takedown';
        
        // Try to capture context
        const takeIdx = text.indexOf('image_output_foundational3p0_takedown');
        const context = text.slice(Math.max(0, takeIdx - 100), Math.min(text.length, takeIdx + 100));
        console.log('[TM Gemini DL] TAKEDOWN DETECTED. Context:', context);
    }
  }

  function scanFetchBody(res, urlStr) {
    state.scanAttempts += 1;
    state.lastStreamUrl = urlStr;
    updateOverlay();

    try {
      const clone = res.clone();
      if (clone.body && typeof clone.body.getReader === 'function' && typeof w.TextDecoder === 'function') {
        state.streamBodiesScanned += 1;
        const reader = clone.body.getReader();
        const decoder = new w.TextDecoder('utf-8');
        const TAIL_KEEP = 8192;
        let tail = '';
        let chunks = 0;

        const pump = () => {
          reader.read().then(({ done, value }) => {
              if (done) {
                try {
                  const rest = decoder.decode();
                  if (rest) extractFromText(tail + rest, { chunk: true });
                } catch (_) {}
                return;
              }
              chunks += 1;
              try {
                const chunkText = decoder.decode(value, { stream: true });
                if (chunkText) {
                  const windowText = tail + chunkText;
                  tail = windowText.slice(-TAIL_KEEP);
                  extractFromText(windowText, { chunk: true });
                }
              } catch (_) {}
              
              if (chunks < 1000) pump(); // Limit recursion
            }).catch((e) => {
              state.scanErrors += 1;
              if (DEBUG) console.error('[TM Gemini DL] scanFetchBody stream error:', e);
            });
        };
        pump();
        return;
      }

      clone.text().then((t) => extractFromText(t)).catch(() => {});
    } catch (e) {
      state.scanErrors += 1;
      if (DEBUG) console.error('[TM Gemini DL] scanFetchBody error:', e);
    }
  }

  function getXhrBodyText(xhr) {
    let result = null;
    try {
      if (typeof xhr.responseText === 'string' && xhr.responseText) result = xhr.responseText;
    } catch (e) {
        if (DEBUG) console.log('[TM Gemini DL] getXhrBodyText: responseText failed', e.message);
    }

    if (!result) {
        try {
          if (xhr.responseType === 'json' && xhr.response) result = JSON.stringify(xhr.response);
        } catch (e) {}
    }

    if (!result) {
        try {
          if (typeof xhr.response === 'string' && xhr.response) result = xhr.response;
        } catch (e) {}
    }
    
    if (DEBUG && !result) {
        // console.log('[TM Gemini DL] getXhrBodyText: NO TEXT FOUND', xhr.responseURL);
    }
    return result;
  }

  // -- Patch Logic Refactored --
  function doPatchFetch() {
      try {
        const origFetch = w.fetch;
        if (typeof origFetch === 'function' && !origFetch.__tm_patched) {
          const newFetch = function (...args) {
            let reqUrl = '';
            try {
              const input = args[0];
              if (typeof input === 'string') reqUrl = input;
              else if (input && typeof input.url === 'string') reqUrl = input.url;
              if (reqUrl) reqUrl = String(new URL(reqUrl, String(w.location && w.location.href)));
            } catch (_) {}
    
            return origFetch.apply(w, args).then((res) => {
              try {
                const url = res && res.url ? res.url : reqUrl;
                state.fetchSeen += 1;
                updateOverlay();
                const urlStr = String(url || '');
                // if (isTarget(url)) ... (omitted direct download to focus on scan)
                if (shouldScanUrl(urlStr)) scanFetchBody(res, urlStr);
              } catch (_) {}
              return res;
            });
          };
          newFetch.__tm_patched = true;
          w.fetch = newFetch;
          myFetch = newFetch;
          log('patched fetch');
        } else {
             if (w.fetch && w.fetch.__tm_patched) myFetch = w.fetch;
        }
      } catch (e) {
        log('fetch patch failed', e);
      }
  }

  function doPatchXHR() {
      try {
        const XHR = w.XMLHttpRequest;
        if (XHR && XHR.prototype && !XHR.prototype.__tm_patched) {
          const XHROpen = XHR.prototype.open;
          const XHRSend = XHR.prototype.send;
    
          // Use defineProperty with writable:false to resist Zone.js overwrites
          const patchedOpen = function (method, url, ...rest) {
            this.__tm_url = url;
            try {
              pushRing(state.xhrUrls, `open ${String(method || '')} ${String(url || '')}`, 50);
            } catch (_) {}
            return XHROpen.call(this, method, url, ...rest);
          };
          
          try {
              Object.defineProperty(XHR.prototype, 'open', {
                  value: patchedOpen,
                  writable: false, 
                  configurable: true 
              });
          } catch(e) {
              XHR.prototype.open = patchedOpen;
          }
    
          const patchedSend = function (body) {
            const self = this;
            
            // Incremental scan for streaming XHR
            this.addEventListener('readystatechange', function () {
              try {
                if (self.readyState !== 3 && self.readyState !== 4) return;
                const url = self.responseURL || self.__tm_url || '';
                const urlStr = String(url || '');
                if (!shouldScanUrl(urlStr)) return;
    
                let text = null;
                try {
                  text = self.responseText;
                } catch (_) {}
                
                if (!text && typeof self.response === 'string') {
                   text = self.response;
                }
                if (typeof text !== 'string' || text.length === 0) return;
    
                const prevLen = self.__tm_scan_len || 0;
                if (text.length <= prevLen) return;
                self.__tm_scan_len = text.length;
    
                let windowText = '';
                // BIGGER WINDOW FIX: Capture first 2MB to ensure we don't miss early payload
                if (!self.__tm_scan_head_done) {
                  self.__tm_scan_head_done = true;
                  windowText = text.slice(0, 2 * 1024 * 1024); 
                } else {
                  const tail = self.__tm_scan_tail || '';
                  const chunk = text.slice(Math.max(0, prevLen - 0));
                  windowText = (tail + chunk).slice(-8192 - 20000);
                  self.__tm_scan_tail = windowText.slice(-8192);
                }
    
                if (!self.__tm_scan_started) {
                  self.__tm_scan_started = true;
                  state.scanAttempts += 1;
                  state.lastStreamUrl = urlStr;
                  pushRing(state.scanEvents, { t: Date.now(), kind: 'xhr', stage: 'stream-start', url: urlStr }, 30);
                  updateOverlay();
                }
    
                extractFromText(windowText, { chunk: true });
              } catch (e) {
                state.scanErrors += 1;
                console.error('[TM Gemini DL] XHR stream scan error:', e);
              }
            });
    
            this.addEventListener('load', function () {
              try {
                state.xhrSeen += 1;
                updateOverlay();
                const bodyText = getXhrBodyText(self);
                if (bodyText) {
                   extractFromText(bodyText);
                }
              } catch (_) {}
            });
    
            return XHRSend.call(this, body);
          };
          
          try {
              Object.defineProperty(XHR.prototype, 'send', {
                  value: patchedSend,
                  writable: false,
                  configurable: true
              });
          } catch(e) {
              XHR.prototype.send = patchedSend;
          }
          
          XHR.prototype.__tm_patched = true;
          myXHR = w.XMLHttpRequest;
          log('patched XHR (locked)');
        } else {
             if (XHR && XHR.prototype && XHR.prototype.__tm_patched) myXHR = w.XMLHttpRequest;
        }
      } catch (e) {
        log('XHR patch failed', e);
      }
  }

  // Initial patch
  doPatchFetch();
  doPatchXHR();
  
  // -- Setter Trap: Catch who is overwriting our patches --
  let fetchTrapActive = false;
  function installFetchTrap() {
      if (fetchTrapActive) return;
      try {
          let currentFetch = w.fetch;
          Object.defineProperty(w, 'fetch', {
              configurable: true,
              enumerable: true,
              get: function() { return currentFetch; },
              set: function(newVal) {
                  const wasPatched = currentFetch && currentFetch.__tm_patched;
                  const isPatched = newVal && newVal.__tm_patched;
                  if (wasPatched && !isPatched) {
                      log('!!! FETCH OVERWRITTEN !!!', { stack: new Error().stack });
                  }
                  currentFetch = newVal;
              }
          });
          fetchTrapActive = true;
      } catch(e) {}
  }
  w.setTimeout(installFetchTrap, 100);

  // Fallbacks
  try {
    if (typeof w.PerformanceObserver === 'function') {
      const obs = new w.PerformanceObserver((list) => {
        for (const e of list.getEntries()) {
          try {
             if (e && typeof e.name === 'string' && isTarget(e.name)) {
               scheduleDownload(normalizeUrl(e.name), undefined);
             }
          } catch (_) {}
        }
      });
      obs.observe({ entryTypes: ['resource'] });
      log('installed PerformanceObserver');
    }
  } catch (e) {}

  function ensurePatches() {
      try {
          const currentFetchPatched = w.fetch && w.fetch.__tm_patched;
          const currentXHRPatched = w.XMLHttpRequest && w.XMLHttpRequest.prototype && w.XMLHttpRequest.prototype.__tm_patched;
          
          if (!currentFetchPatched && typeof w.fetch === 'function') {
              // log('fetch lost patch! Re-patching...');
              doPatchFetch();
          }
          
          if (!currentXHRPatched && w.XMLHttpRequest && w.XMLHttpRequest.prototype) {
              // log('XHR lost patch! Re-patching...');
              doPatchXHR();
          }
      } catch(e) {}
  }
  
  w.setInterval(ensurePatches, 1000);
  
  // -- SELF TEST ON LOAD --

  
  // Expose manual re-patch
  w.__tm_repatch = function() {
      doPatchFetch();
      doPatchXHR();
      updateOverlay();
  };

  log('installed', { href: (() => { try { return String(w.location && w.location.href); } catch (_) { return null; } })() });

})();
