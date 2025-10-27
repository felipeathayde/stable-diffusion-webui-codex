"use strict";
// @ts-check
/*
 DevNotes (2025-10-12)
 - Purpose: Busca/filtra itens de Settings e mostra/oculta abas auxiliares.
 - Safety: Guards para elementos nulos; getAppElementById para IDs; onEdit com debounce.
*/

/** @type {Record<string, number>} */
let settingsExcludeTabsFromShowAll = {
    settings_tab_defaults: 1,
    settings_tab_sysinfo: 1,
    settings_tab_actions: 1,
    settings_tab_licenses: 1,
};

/**
 * @param {string} id
 */
function getAppElementById(id) {
    const root = gradioApp();
    if ('getElementById' in root && typeof root.getElementById === 'function') {
        const el = root.getElementById(id);
        if (el instanceof HTMLElement) return el;
    }
    const fallback = document.getElementById(id);
    return fallback instanceof HTMLElement ? fallback : null;
}

function settingsShowAllTabs() {
    gradioApp().querySelectorAll('#settings > div').forEach(function(elem) {
        if (!(elem instanceof HTMLElement)) return;
        if (settingsExcludeTabsFromShowAll[elem.id]) return;
        elem.style.display = 'block';
    });
}

function settingsShowOneTab() {
    const btn = getAppElementById('settings_show_one_page');
    if (btn) btn.click();
}

function attachSettingsSearch() {
    var root = gradioApp();
    var edit = root.querySelector('#settings_search');
    var editTextarea = root.querySelector('#settings_search > label > input');
    var buttonShowAllPages = getAppElementById('settings_show_all_pages');
    var settings_tabs = root.querySelector('#settings div');

    if (!edit || !editTextarea || !buttonShowAllPages || !settings_tabs) {
        return;
    }

    if (edit instanceof HTMLElement && edit.dataset.bound === 'true') {
        return;
    }
    if (edit instanceof HTMLElement) edit.dataset.bound = 'true';

    onEdit('settingsSearch', /** @type {HTMLInputElement} */ (editTextarea), 200, function() {
        var searchText = ((/** @type {HTMLInputElement} */ (editTextarea)).value || '').trim().toLowerCase();

        // Hide/show individual entries
        root.querySelectorAll('#settings > div[id^=settings_] div[id^=column_settings_] > *').forEach(function(elem) {
            if (!(elem instanceof HTMLElement)) return;
            var visible = (elem.textContent || '').trim().toLowerCase().indexOf(searchText) !== -1;
            elem.style.display = visible ? '' : 'none';
        });

        // Hide entire section columns with no visible children; show sections with matches
        root.querySelectorAll('#settings > div[id^=settings_] div[id^=column_settings_]').forEach(function(col) {
            if (!(col instanceof HTMLElement)) return;
            var anyVisible = Array.from(col.children).some(function(child) {
                return child instanceof HTMLElement && child.style.display !== 'none';
            });
            col.style.display = anyVisible || searchText === '' ? '' : 'none';
        });

        // If searching, expand all tabs; otherwise revert to single tab
        if (searchText !== "") {
            settingsShowAllTabs();
            // Scroll to first visible match for convenience
            var first = root.querySelector('#settings > div[id^=settings_] div[id^=column_settings_] > *:not([style*="display: none"])');
            if (first && first instanceof HTMLElement) {
                try { first.scrollIntoView({behavior:'smooth', block:'center'}); } catch(e) {}
            }
        } else {
            settingsShowOneTab();
        }
    });

    if (settings_tabs && settings_tabs.firstChild !== edit) {
        settings_tabs.insertBefore(edit, settings_tabs.firstChild || null);
    }
    if (buttonShowAllPages && !buttonShowAllPages.dataset.bound) {
        buttonShowAllPages.addEventListener('click', settingsShowAllTabs);
        buttonShowAllPages.dataset.bound = 'true';
    }
    if (settings_tabs && settings_tabs.lastChild !== buttonShowAllPages) {
        settings_tabs.appendChild(buttonShowAllPages);
    }
}

onUiLoaded(attachSettingsSearch);
onUiUpdate(attachSettingsSearch);


onOptionsChanged(function() {
    if (gradioApp().querySelector('#settings .settings-category')) return;

    /** @type {Record<string, HTMLElement>} */
    var sectionMap = {};
    gradioApp().querySelectorAll('#settings > div > button').forEach(function(x) {
        if (!(x instanceof HTMLElement)) return;
        sectionMap[(x.textContent || '').trim()] = x;
    });

    const categories = Array.isArray(opts._categories) ? opts._categories : [];
    categories.forEach(
        /** @param {unknown} entry */
        (entry) => {
            if (!Array.isArray(entry) || entry.length < 2) return;
            const [sectionKey, categoryKey] = entry;
            const localizationDict = typeof window.localization === 'object' && window.localization ? window.localization : undefined;
            const section = (localizationDict && typeof localizationDict[sectionKey] === 'string') ? localizationDict[sectionKey] : sectionKey;
            const category = (localizationDict && typeof localizationDict[categoryKey] === 'string') ? localizationDict[categoryKey] : categoryKey;

            var span = document.createElement('SPAN');
            span.textContent = category;
            span.className = 'settings-category';

            var sectionElem = sectionMap[section];
            if (!sectionElem) return;
            if (sectionElem.parentElement) {
                sectionElem.parentElement.insertBefore(span, sectionElem);
            }
        }
    );
});
