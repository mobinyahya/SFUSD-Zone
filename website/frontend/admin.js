// SFUSD Admin Console - Admin Page (Filter Interface)
// Depends on shared.js being loaded first.

let authToken = null;

// Filter state
let solutionSpaceStats = {};
let filterBounds = {};
let visibleMetrics = new Set();
let totalPareto = 0;
let currentFilteredCount = 0;
let feasibleStats = {};
let availableStats = {};
let categories = {};

// Debounce
let filterTimer = null;
const FILTER_DEBOUNCE_MS = 300;

// ============================================================================
// Page Hooks (configure shared.js behavior for admin page)
// ============================================================================

pageHooks.rightPanelSelector = '#right-panel';

// ============================================================================
// Auth
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('auth-submit').addEventListener('click', authenticate);
    document.getElementById('auth-password').addEventListener('keypress', e => {
        if (e.key === 'Enter') authenticate();
    });
});

async function authenticate() {
    const password = document.getElementById('auth-password').value;
    try {
        const res = await fetch(`${API_BASE}/api/admin/auth`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ password }),
        });
        if (!res.ok) {
            document.getElementById('auth-error').classList.remove('hidden');
            return;
        }
        const data = await res.json();
        authToken = data.token;
        sessionStorage.setItem('admin_token', authToken);
        document.getElementById('auth-gate').classList.add('hidden');
        document.getElementById('admin-app').classList.remove('hidden');
        initAdmin();
    } catch (err) {
        document.getElementById('auth-error').classList.remove('hidden');
    }
}

function adminFetch(url, opts = {}) {
    opts.headers = { ...(opts.headers || {}), 'Authorization': `Bearer ${authToken}` };
    return fetch(url, opts);
}

// ============================================================================
// Init
// ============================================================================

async function initAdmin() {
    await fetchMetricsConfig();
    initMap();
    setupResizeHandle();
    setupEventListeners();
    await loadSolutionSpace();
    applyFilters();
}

// ============================================================================
// Event Listeners
// ============================================================================

function setupEventListeners() {
    document.getElementById('reset-all-btn').addEventListener('click', () => {
        for (const col of Object.keys(filterBounds)) {
            filterBounds[col] = { min_bound: null, max_bound: null };
        }
        renderFilterSliders();
        applyFilters();
    });

    document.getElementById('save-solution-btn').addEventListener('click', saveSolution);
    setupMetricSearch();
    setupChartsClose();
}

// ============================================================================
// Solution Space & Filter Logic
// ============================================================================

async function loadSolutionSpace() {
    const res = await adminFetch(`${API_BASE}/api/admin/solution-space`);
    const data = await res.json();
    solutionSpaceStats = data.metrics;
    totalPareto = data.total_pareto;
    categories = data.categories;

    filterBounds = {};
    for (const col of Object.keys(solutionSpaceStats)) {
        filterBounds[col] = { min_bound: null, max_bound: null };
    }

    visibleMetrics = new Set();
    for (const [col, stat] of Object.entries(solutionSpaceStats)) {
        if (stat.is_core) visibleMetrics.add(col);
    }

    renderFilterSliders();
    updateSolutionBadge(totalPareto, totalPareto);
}

function updateSolutionBadge(filtered, total) {
    currentFilteredCount = filtered;
    document.getElementById('solution-count-badge').textContent = `${filtered} / ${total} Pareto solutions`;
}

async function applyFilters() {
    const activeBounds = {};
    for (const col of Object.keys(filterBounds)) {
        const b = filterBounds[col];
        if (b.min_bound != null || b.max_bound != null) activeBounds[col] = b;
    }

    const res = await adminFetch(`${API_BASE}/api/admin/filter`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bounds: activeBounds }),
    });
    const data = await res.json();

    updateSolutionBadge(data.solution_count, data.total_pareto);
    feasibleStats = data.feasible_stats || {};
    availableStats = data.available_stats || {};
    adjustSliderRanges();

    if (data.solution_count === 0) {
        showZeroState();
        updateCentroidSummary(null);
    } else {
        hideZeroState();
        updateCentroidSummary(data.centroid_metrics);
        if (data.centroid_path) loadSolution(data.centroid_path);
    }
}

function debouncedApplyFilters() {
    clearTimeout(filterTimer);
    filterTimer = setTimeout(() => applyFilters(), FILTER_DEBOUNCE_MS);
}

// ============================================================================
// Zero-state (no feasible solutions)
// ============================================================================

function showZeroState() {
    const alert = document.getElementById('zero-state-alert');
    alert.classList.remove('hidden');

    const activeBounds = {};
    for (const col of Object.keys(filterBounds)) {
        const b = filterBounds[col];
        if (b.min_bound != null || b.max_bound != null) activeBounds[col] = b;
    }

    adminFetch(`${API_BASE}/api/admin/suggest-relaxation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bounds: activeBounds }),
    }).then(r => r.json()).then(data => {
        const container = document.getElementById('relaxation-suggestions');
        const suggestions = data.suggestions || {};
        if (Object.keys(suggestions).length === 0) {
            container.innerHTML = '<em>No single-metric relaxation found. Try resetting.</em>';
            return;
        }
        container.innerHTML = '';
        for (const [col, newBound] of Object.entries(suggestions)) {
            const stat = solutionSpaceStats[col];
            if (!stat) continue;
            const btn = document.createElement('button');
            btn.className = 'relaxation-btn';
            btn.textContent = `Relax ${stat.display_name} to ${formatMetricVal(col, newBound)}`;
            btn.addEventListener('click', () => {
                if (stat.direction === 'minimize') filterBounds[col].max_bound = newBound;
                else filterBounds[col].min_bound = newBound;
                updateSliderUI(col);
                applyFilters();
            });
            container.appendChild(btn);
        }
    });
}

function hideZeroState() {
    document.getElementById('zero-state-alert').classList.add('hidden');
}

// ============================================================================
// Centroid Summary
// ============================================================================

function updateCentroidSummary(metrics) {
    const panel = document.getElementById('centroid-summary');
    const container = document.getElementById('centroid-metrics');
    if (!metrics) { panel.classList.add('hidden'); return; }
    panel.classList.remove('hidden');

    let html = '';
    for (const [col, stat] of Object.entries(solutionSpaceStats)) {
        if (!stat.is_core || !(col in metrics)) continue;
        html += `<div class="centroid-metric">
            <span class="centroid-metric-name">${stat.short_name || stat.display_name}</span>
            <span class="centroid-metric-value">${formatMetricVal(col, metrics[col])}</span>
        </div>`;
    }
    container.innerHTML = html;
}

// ============================================================================
// Filter Sliders
// ============================================================================

function renderFilterSliders() {
    const container = document.getElementById('filter-sliders');
    container.innerHTML = '';

    const grouped = {};
    for (const col of visibleMetrics) {
        const stat = solutionSpaceStats[col];
        if (!stat) continue;
        const cat = stat.category;
        if (!grouped[cat]) grouped[cat] = [];
        grouped[cat].push(col);
    }

    const catOrder = ['diversity', 'distance', 'programs', 'quality'];
    for (const cat of catOrder) {
        const cols = grouped[cat];
        if (!cols || !cols.length) continue;

        const catDiv = document.createElement('div');
        catDiv.className = 'filter-category';
        catDiv.dataset.category = cat;

        const catLabel = categories[cat] || cat;
        catDiv.innerHTML = `
            <div class="filter-category-header" data-cat="${cat}">
                <span class="filter-category-title"><span class="chevron">&#9654;</span> ${catLabel}</span>
                <span class="filter-category-count">${cols.length} metrics</span>
            </div>
            <div class="filter-category-metrics collapsed" data-cat="${cat}"></div>
        `;

        const metricsDiv = catDiv.querySelector('.filter-category-metrics');
        cols.forEach(col => metricsDiv.appendChild(buildSliderRow(col)));

        catDiv.querySelector('.filter-category-header').addEventListener('click', () => {
            const m = catDiv.querySelector('.filter-category-metrics');
            const chevron = catDiv.querySelector('.chevron');
            if (m.classList.contains('collapsed')) {
                m.classList.remove('collapsed');
                chevron.innerHTML = '&#9660;';
            } else {
                m.classList.add('collapsed');
                chevron.innerHTML = '&#9654;';
            }
        });

        container.appendChild(catDiv);
    }
}

function buildSliderRow(col) {
    const stat = solutionSpaceStats[col];
    const gMin = stat.min;
    const gMax = stat.max;
    const dir = stat.direction === 'minimize' ? 'lower is better' : 'higher is better';
    const isCore = stat.is_core;
    const step = getStep(gMin, gMax);

    const row = document.createElement('div');
    row.className = 'metric-slider-row';
    row.dataset.col = col;

    const b = filterBounds[col] || {};
    const lo = b.min_bound != null ? b.min_bound : gMin;
    const hi = b.max_bound != null ? b.max_bound : gMax;
    const hasFilter = b.min_bound != null || b.max_bound != null;

    row.innerHTML = `
        <div class="metric-slider-header">
            <span class="metric-slider-name" title="${stat.description}">${stat.display_name}</span>
            <span class="metric-slider-direction">${dir}</span>
            <div class="metric-slider-actions">
                ${hasFilter ? '<button class="btn-loosen" data-action="loosen" title="Loosen this filter">Loosen</button>' : ''}
                <button data-action="reset" title="Reset this filter">Reset</button>
                ${!isCore ? '<button class="btn-remove" data-action="remove" title="Remove this metric">x</button>' : ''}
            </div>
        </div>
        <div class="slider-track-container">
            <div class="slider-track">
                <div class="slider-feasible-band"></div>
                <div class="slider-active-band"></div>
            </div>
            <input type="range" class="slider-min" min="${gMin}" max="${gMax}" step="${step}" value="${lo}">
            <input type="range" class="slider-max" min="${gMin}" max="${gMax}" step="${step}" value="${hi}">
        </div>
        <div class="slider-values">
            <span class="slider-bound-value sv-lo">${formatMetricVal(col, lo)}</span>
            <span class="slider-bound-value sv-hi">${formatMetricVal(col, hi)}</span>
        </div>
    `;

    const minSlider = row.querySelector('.slider-min');
    const maxSlider = row.querySelector('.slider-max');

    const onSliderChange = () => {
        let loVal = parseFloat(minSlider.value);
        let hiVal = parseFloat(maxSlider.value);
        if (loVal > hiVal) { loVal = hiVal; minSlider.value = loVal; }

        const isMinChanged = Math.abs(loVal - gMin) > step * 0.5;
        const isMaxChanged = Math.abs(hiVal - gMax) > step * 0.5;
        filterBounds[col] = {
            min_bound: isMinChanged ? loVal : null,
            max_bound: isMaxChanged ? hiVal : null,
        };

        updateSliderVisuals(row, col);
        debouncedApplyFilters();
    };

    minSlider.addEventListener('input', onSliderChange);
    maxSlider.addEventListener('input', onSliderChange);

    row.querySelectorAll('[data-action]').forEach(btn => {
        btn.addEventListener('click', e => {
            e.stopPropagation();
            const action = btn.dataset.action;
            if (action === 'reset') {
                filterBounds[col] = { min_bound: null, max_bound: null };
                minSlider.value = gMin;
                maxSlider.value = gMax;
                updateSliderVisuals(row, col);
                applyFilters();
            } else if (action === 'remove') {
                visibleMetrics.delete(col);
                filterBounds[col] = { min_bound: null, max_bound: null };
                renderFilterSliders();
                applyFilters();
            } else if (action === 'loosen') {
                loosenMetric(col);
            }
        });
    });

    updateSliderVisuals(row, col);
    return row;
}

function updateSliderVisuals(row, col) {
    const stat = solutionSpaceStats[col];
    const gMin = stat.min;
    const gMax = stat.max;
    const range = gMax - gMin || 1;

    const minSlider = row.querySelector('.slider-min');
    const maxSlider = row.querySelector('.slider-max');
    const lo = parseFloat(minSlider.value);
    const hi = parseFloat(maxSlider.value);

    const activeBand = row.querySelector('.slider-active-band');
    const leftPct = ((lo - gMin) / range) * 100;
    const widthPct = ((hi - lo) / range) * 100;
    activeBand.style.left = leftPct + '%';
    activeBand.style.width = widthPct + '%';

    const fBand = row.querySelector('.slider-feasible-band');
    const fs = feasibleStats[col];
    if (fs) {
        const fLeft = ((fs.min - gMin) / range) * 100;
        const fWidth = ((fs.max - fs.min) / range) * 100;
        fBand.style.left = Math.max(0, fLeft) + '%';
        fBand.style.width = Math.min(100 - fLeft, fWidth) + '%';
    } else {
        fBand.style.left = '0%';
        fBand.style.width = '100%';
    }

    const b = filterBounds[col] || {};
    const loLabel = row.querySelector('.sv-lo');
    const hiLabel = row.querySelector('.sv-hi');
    loLabel.textContent = formatMetricVal(col, lo);
    hiLabel.textContent = formatMetricVal(col, hi);
    loLabel.classList.toggle('active', b.min_bound != null);
    hiLabel.classList.toggle('active', b.max_bound != null);

    const hasFilter = b.min_bound != null || b.max_bound != null;
    const actions = row.querySelector('.metric-slider-actions');
    let loosenBtn = actions.querySelector('.btn-loosen');
    if (hasFilter && !loosenBtn) {
        loosenBtn = document.createElement('button');
        loosenBtn.className = 'btn-loosen';
        loosenBtn.dataset.action = 'loosen';
        loosenBtn.title = 'Loosen this filter';
        loosenBtn.textContent = 'Loosen';
        loosenBtn.addEventListener('click', e => { e.stopPropagation(); loosenMetric(col); });
        actions.insertBefore(loosenBtn, actions.firstChild);
    } else if (!hasFilter && loosenBtn) {
        loosenBtn.remove();
    }
}

function updateSliderUI(col) {
    const row = document.querySelector(`.metric-slider-row[data-col="${col}"]`);
    if (!row) return;
    const stat = solutionSpaceStats[col];
    const b = filterBounds[col] || {};
    row.querySelector('.slider-min').value = b.min_bound != null ? b.min_bound : stat.min;
    row.querySelector('.slider-max').value = b.max_bound != null ? b.max_bound : stat.max;
    updateSliderVisuals(row, col);
}

function adjustSliderRanges() {
    if (Object.keys(feasibleStats).length === 0) return;
    for (const col of visibleMetrics) {
        const row = document.querySelector(`.metric-slider-row[data-col="${col}"]`);
        if (!row) continue;
        updateSliderVisuals(row, col);
    }
}

function loosenMetric(col) {
    const stat = solutionSpaceStats[col];
    const b = filterBounds[col];
    if (!b) return;

    const available = availableStats[col];
    const expansionFactor = 0.3;
    const step = getStep(stat.min, stat.max);

    if (stat.direction === 'minimize' && b.max_bound != null) {
        const target = available ? available.max : stat.max;
        if (Math.abs(b.max_bound - target) < step * 0.5) return;
        b.max_bound = Math.min(target, b.max_bound + (target - b.max_bound) * expansionFactor);
        if (Math.abs(b.max_bound - target) < step * 0.5) b.max_bound = target;
    } else if (stat.direction === 'maximize' && b.min_bound != null) {
        const target = available ? available.min : stat.min;
        if (Math.abs(b.min_bound - target) < step * 0.5) return;
        b.min_bound = Math.max(target, b.min_bound - (b.min_bound - target) * expansionFactor);
        if (Math.abs(b.min_bound - target) < step * 0.5) b.min_bound = target;
    }

    updateSliderUI(col);
    applyFilters();
}

// ============================================================================
// Metric Search
// ============================================================================

function setupMetricSearch() {
    const input = document.getElementById('metric-search');
    const results = document.getElementById('metric-search-results');

    input.addEventListener('input', () => {
        const q = input.value.trim().toLowerCase();
        if (!q) { results.classList.add('hidden'); return; }

        const matches = [];
        for (const [col, stat] of Object.entries(solutionSpaceStats)) {
            if (stat.display_name.toLowerCase().includes(q) || stat.description.toLowerCase().includes(q)) {
                matches.push({ col, stat });
            }
        }

        if (matches.length === 0) { results.classList.add('hidden'); return; }

        results.innerHTML = '';
        matches.slice(0, 8).forEach(({ col, stat }) => {
            const already = visibleMetrics.has(col);
            const div = document.createElement('div');
            div.className = 'search-result-item' + (already ? ' already-added' : '');
            div.innerHTML = `<span>${stat.display_name}</span><span class="search-result-cat">${stat.category}</span>`;
            if (!already) {
                div.addEventListener('click', () => {
                    visibleMetrics.add(col);
                    input.value = '';
                    results.classList.add('hidden');
                    renderFilterSliders();
                });
            }
            results.appendChild(div);
        });
        results.classList.remove('hidden');
    });

    input.addEventListener('blur', () => {
        setTimeout(() => results.classList.add('hidden'), 200);
    });
}

// ============================================================================
// Solution History (admin version)
// ============================================================================

function saveSolution() {
    if (!currentSolution) return;
    const path = currentSolution.path || '';
    if (path && savedSolutions.some(s => s.path === path)) return;
    if (savedSolutions.length >= MAX_SAVED_SOLUTIONS) {
        savedSolutions.shift();
        savedSolutions.forEach((s, i) => { s.index = i + 1; });
    }

    const index = savedSolutions.length + 1;
    const categoryScores = currentSolution.category_percentiles
        ? { ...currentSolution.category_percentiles }
        : {};

    savedSolutions.push({
        index,
        path,
        solutionData: JSON.parse(JSON.stringify(currentSolution)),
        label: `Solution #${index}`,
        pros: '',
        cons: '',
        timestamp: new Date().toISOString(),
        categoryScores,
    });
    currentViewedIndex = index;
    renderSolutionHistory();
}

// ============================================================================
// Utilities
// ============================================================================

function formatMetricVal(col, value) {
    if (value == null) return '-';
    const stat = solutionSpaceStats[col];
    if (!stat) return value.toFixed(3);
    const range = stat.max - stat.min;
    if (range > 100) return Math.round(value).toString();
    if (range > 1) return value.toFixed(2);
    return value.toFixed(4);
}

function getStep(min, max) {
    const range = max - min;
    if (range === 0) return 0.001;
    if (range > 100) return 1;
    if (range > 10) return 0.1;
    if (range > 1) return 0.01;
    return 0.0001;
}
