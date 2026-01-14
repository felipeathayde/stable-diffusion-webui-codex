// ==UserScript==
// @name         Gemini: Safety Diagnostics & Overlay
// @namespace    local
// @version      0.1.0
// @description  Monitors Gemini safety classifiers (refusal reasons) and displays them in an overlay. Does NOT download images.
// @match        https://gemini.google.com/*
// @run-at       document-start
// @connect      lh3.googleusercontent.com
// @connect      *.googleusercontent.com
// ==/UserScript==


(function () {
  'use strict';

  const DEBUG = true;
  // When true, automatically clicks the "Refazer" (retry) button when a refusal takedown is detected
  const AUTO_RETRY_ON_REFUSAL = true;
  
  const w = typeof unsafeWindow !== 'undefined' ? unsafeWindow : window;
  const pageConsole = w && w.console && typeof w.console.log === 'function' ? w.console : console;
  
  if (w.__tm_gemini_safety_installed) return;
  w.__tm_gemini_safety_installed = true;

  const state = w.__tm_gemini_safety_state || (w.__tm_gemini_safety_state = {
      installedAt: new Date().toISOString(),
      fetchSeen: 0,
      xhrSeen: 0,
      scanAttempts: 0,
      scanErrors: 0,
      metrics: {
          takedowns: 0,
          lastRefusal: null,
          safetyScores: []
      }
  });
    
  // -- Helper Logger --
  function log(...args) {
    if (DEBUG) pageConsole.log('[TM Gemini Safety]', ...args);
  }

  // -- Auto-Retry on Refusal --
  // Selector for the regenerate/retry button
  const RETRY_BUTTON_SELECTOR = 'regenerate-button button[aria-label="Refazer"], regenerate-button button mat-icon[fonticon="refresh"]';
  
  // Track if we're waiting for the button to appear
  let pendingRetryClick = false;
  let retryObserver = null;
  
  // Check if button is actually clickable (not disabled, not invisible)
  function isButtonClickable(button) {
    if (!button) return false;
    
    // Check disabled attribute
    if (button.disabled) return false;
    if (button.getAttribute('aria-disabled') === 'true') return false;
    
    // Check visibility
    const style = window.getComputedStyle(button);
    if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;
    
    // Check if element is in the DOM and has dimensions
    const rect = button.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return false;
    
    return true;
  }
  
  function findAndClickRetryButton() {
    const retryButton = document.querySelector(RETRY_BUTTON_SELECTOR);
    if (retryButton) {
      // If we found the icon, get the parent button
      const button = retryButton.closest('button') || retryButton;
      
      // Check if button is actually clickable
      if (!isButtonClickable(button)) {
        log('Retry button found but not clickable yet (disabled/hidden)');
        return false; // Keep waiting
      }
      
      log('Found clickable retry button, clicking...');
      // Small delay to ensure UI is ready
      setTimeout(() => {
        button.click();
        pendingRetryClick = false;
      }, 300);
      return true;
    }
    return false;
  }
  
  function setupRetryObserver() {
    if (retryObserver) return; // Already set up
    
    retryObserver = new MutationObserver((mutations) => {
      if (!pendingRetryClick) return;
      
      // Check on any mutation (childList or attributes)
      // Try to find and click the button
      findAndClickRetryButton();
    });
    
    // Wait for a valid root element to observe
    function startObserving() {
      const root = document.body || document.documentElement;
      if (!root) {
        // Neither body nor documentElement exist, wait and retry
        log('No root element yet, waiting...');
        setTimeout(startObserving, 100);
        return;
      }
      
      // Observe for both childList changes and attribute changes
      retryObserver.observe(root, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['disabled', 'aria-disabled', 'class', 'style', 'hidden']
      });
      
      log('Retry button observer installed on', root.nodeName);
      
      // Do an immediate check in case button is already there
      findAndClickRetryButton();
    }
    
    startObserving();
  }
  
  function clickRetryButton() {
    if (!AUTO_RETRY_ON_REFUSAL) return;
    
    // Try to find and click immediately
    if (findAndClickRetryButton()) {
      log('Auto-clicked retry button immediately');
      return;
    }
    
    // Button not found or not clickable yet, set up observer to wait
    log('Retry button not ready, setting up observer to wait...');
    pendingRetryClick = true;
    setupRetryObserver();
  }

  // -- Overlay UI --
  let overlayEl = null;

  function updateOverlay() {
      if (!overlayEl) return;
      
      const lines = [];
      lines.push('Gemini Safety Monitor');
      lines.push(`Fetch: ${state.fetchSeen} | XHR: ${state.xhrSeen}`);
      lines.push(`Scans: ${state.scanAttempts}`);
    
      if (state.metrics.takedowns > 0) {
          lines.push('--- TAKEDOWNS ---');
          lines.push('Count: ' + state.metrics.takedowns);
          if (state.metrics.lastRefusal) {
              lines.push('Refusal: ' + state.metrics.lastRefusal.slice(0, 30));
          }
      }

      if (state.metrics.safetyScores.length > 0) {
          lines.push('--- CLASSIFIERS ---');
          state.metrics.safetyScores.forEach(item => {
               // Colorize high scores
               const scoreStr = item.score.toFixed(4);
               lines.push(`${item.label}: ${scoreStr}`);
          });
      } else {
          lines.push('--- Waiting for Data ---');
      }
      
      overlayEl.textContent = lines.join('\n');
  }
  
  function createOverlay() {
      if (document.getElementById('tm-gemini-safety-overlay')) return;
      const container = document.createElement('div');
      container.id = 'tm-gemini-safety-overlay';
      container.style.position = 'fixed';
      container.style.bottom = '10px';
      container.style.right = '10px';
      container.style.zIndex = '999999';
      container.style.border = '2px solid #f00'; // Red border for Safety Monitor
      container.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
      container.style.color = '#fff';
      container.style.padding = '8px';
      container.style.fontFamily = 'monospace';
      container.style.fontSize = '12px';
      container.style.lineHeight = '1.3';
      container.style.pointerEvents = 'none'; // Passthrough clicks
      
      const statsDiv = document.createElement('div');
      statsDiv.style.whiteSpace = 'pre';
      container.appendChild(statsDiv);
      overlayEl = statsDiv;
      
      document.body.appendChild(container);
      updateOverlay();
  }
  
  if (document.body) {
      createOverlay();
  } else {
      document.addEventListener('DOMContentLoaded', createOverlay);
  }

  // -- Text Extraction (Classifier Focused) --
  function extractClassifiers(text) {
    if (!text || typeof text !== 'string') return;
    
    let foundAny = false;

    // 1. Classifier Scores extraction
    // Robust approach: Find key, grab window, regex for pairs [ "key" , value ]
    const keysOfInterest = [
        'grail_t2i_image_output_classifier',
        'image_and_video_safety_classifier',
        'text_safety_classifier',
        'csam_output_safety',
        'merged_image_output_3p0', // For IDENTIFIABLE_PEOPLE
        'prominent_people_3p0',    // For Celebrities
        'image_upload_classifier',
        'child_safety_mmgen_classifier'
    ];

    for (const mainKey of keysOfInterest) {
        const keyIdx = text.indexOf(mainKey);
        if (keyIdx === -1) continue;

        if (DEBUG) console.log(`[TM Gemini Safety] Found key: ${mainKey}`);

        const windowSize = 4000; 
        const snippet = text.slice(keyIdx, keyIdx + windowSize);

        // Regex to find: [ "label" , 0.123 ] 
        // Label: Now handles UPPERCASE/lowercase/underscores.
        const pairRegex = /\[\\*?"([a-zA-Z0-9_]+)\\*?"\s*,\s*(\d+(?:\.\d+)?(?:E-?\d+)?)\s*\]/g;
        
        let m;
        while ((m = pairRegex.exec(snippet)) !== null) {
            const key = m[1];
            const valStr = m[2];
            const score = parseFloat(valStr);

            // Filter logic:
            // - Standard safety keys (porn, csam, etc)
            // - Specific uppercase violations (VIOLATES_...)
            // - "image_upload_..."
            const isRelevant = 
                key.includes('image_output') || 
                key.includes('csam') || 
                key.includes('porn') || 
                key.includes('pedo') || 
                key.includes('hate') ||
                key.includes('harassment') ||
                key.includes('violence') ||
                key.includes('dangerous') ||
                key.startsWith('VIOLATES_') ||
                key.includes('image_upload') ||
                key.includes('prominent_people');

            if (isRelevant) {
                const existing = state.metrics.safetyScores.find(s => s.label === key);
                if (existing) {
                    existing.score = score;
                } else {
                    state.metrics.safetyScores.push({ label: key, score: score });
                }
                foundAny = true;
            }
        }
    }
    
    if (foundAny) {
        // Sort Descending
        state.metrics.safetyScores.sort((a, b) => b.score - a.score);
        // Keep top 12 (expanded list)
        if (state.metrics.safetyScores.length > 12) {
            state.metrics.safetyScores.length = 12;
        }
        updateOverlay();
    }
    
    // 2. Takedown keywords
    if (text.includes('image_output_foundational3p0_takedown')) {
        state.metrics.takedowns += 1;
        state.metrics.lastRefusal = 'Foundational Takedown';
        updateOverlay();
        console.warn('[TM Gemini Safety] TAKEDOWN DETECTED: Foundational');
        clickRetryButton();
    }
    if (text.includes('image_output_identifiable_people_3p0_takedown')) {
        state.metrics.takedowns += 1;
        state.metrics.lastRefusal = 'Identifiable People Takedown';
        updateOverlay();
        console.warn('[TM Gemini Safety] TAKEDOWN DETECTED: Identifiable People');
        clickRetryButton();
    }
  }

  // -- Network Scanning (Read Only) --
  function shouldScanUrl(url) {
    const u = String(url || '');
    return /BardChatUi\/data\//i.test(u) || /StreamGenerate|batchexecute/i.test(u);
  }

  function scanFetchBody(res) {
    state.scanAttempts += 1;
    updateOverlay();
    try {
      const clone = res.clone();
      if (clone.body && typeof clone.body.getReader === 'function' && typeof w.TextDecoder === 'function') {
        const reader = clone.body.getReader();
        const decoder = new w.TextDecoder('utf-8');
        const pump = () => {
          reader.read().then(({ done, value }) => {
              if (done) return;
              try {
                const chunkText = decoder.decode(value, { stream: true });
                if (chunkText) extractClassifiers(chunkText);
              } catch (_) {}
              pump();
            }).catch(() => {});
        };
        pump();
        return;
      }
      clone.text().then((t) => extractClassifiers(t)).catch(() => {});
    } catch (e) {
      state.scanErrors += 1;
    }
  }

  function doPatchFetch() {
      const origFetch = w.fetch;
      if (!origFetch || origFetch.__tm_patched) return;
      
      const newFetch = function (...args) {
        let reqUrl = '';
        try {
             if (typeof args[0] === 'string') reqUrl = args[0];
             else if (args[0] && args[0].url) reqUrl = args[0].url;
        } catch(_) {}
        
        return origFetch.apply(w, args).then((res) => {
          try {
            const url = res.url || reqUrl;
            state.fetchSeen += 1;
            if (shouldScanUrl(url)) scanFetchBody(res);
          } catch (_) {}
          return res;
        });
      };
      newFetch.__tm_patched = true;
      w.fetch = newFetch;
  }

  function doPatchXHR() {
      const XHR = w.XMLHttpRequest;
      if (!XHR || !XHR.prototype || XHR.prototype.__tm_patched) return;
      
      const XHROpen = XHR.prototype.open;
      const XHRSend = XHR.prototype.send;
      
      XHR.prototype.open = function(method, url, ...rest) {
          this.__tm_url = url;
          return XHROpen.call(this, method, url, ...rest);
      };
      
      XHR.prototype.send = function(body) {
          const self = this;
          this.addEventListener('readystatechange', function() {
              try {
                  if (self.readyState === 3 || self.readyState === 4) {
                      const url = self.responseURL || self.__tm_url || '';
                      if (shouldScanUrl(url)) {
                          let text = self.responseText || self.response;
                          if (typeof text === 'string') extractClassifiers(text);
                      }
                  }
              } catch(e) {}
          });
          
          this.addEventListener('load', function() {
              state.xhrSeen += 1;
              updateOverlay();
          });
          
          return XHRSend.call(this, body);
      };
      
      XHR.prototype.__tm_patched = true;
  }

  // Init
  doPatchFetch();
  doPatchXHR();
  w.setInterval(() => { doPatchFetch(); doPatchXHR(); }, 2000);
  
  log('Safety Monitor Installed');

})();
