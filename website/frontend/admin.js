// SFUSD Admin Console

const API_BASE = '';

// ============================================================================
// State
// ============================================================================

let authToken = null;
let map = null;
let geojsonLayer = null;
let geojsonData = null;
let schoolMarkersLayer = null;
let schoolsVisible = false;
let currentSolution = null;
let singleChart = null;
let selectedMetricKey = null;

// Filter state
let solutionSpaceStats = {};   // {metric_col: {min, max, p25, p50, p75, direction, display_name, ...}}
let filterBounds = {};          // {metric_col: {min_bound, max_bound}} - null = unconstrained
let visibleMetrics = new Set(); // metric columns currently shown as sliders
let totalPareto = 0;
let currentFilteredCount = 0;
let feasibleStats = {};         // {metric_col: {min, max}} from last filter response
let categories = {};

// Solution history
let savedSolutions = [];
let currentViewedIndex = null;
let historyExpanded = false;
const MAX_SAVED_SOLUTIONS = 30;

// Debounce
let filterTimer = null;
const FILTER_DEBOUNCE_MS = 300;

// Chart colors
const CHART_COLORS = {
    primary: '#3498db',
    secondary: '#2ecc71',
    tertiary: '#9b59b6',
    quaternary: '#e74c3c',
    quinary: '#f39c12',
    ethnicities: {
        'Black/African American': '#e74c3c',
        'Hispanic/Latinx': '#f39c12',
        'White': '#3498db',
        'Asian': '#2ecc71',
        'Other': '#9b59b6',
    }
};

const METRIC_CHART_CONFIG = {
    'theil_index': { type: 'ethnicity', title: 'Ethnic Composition by Zone' },
    'FRL': { type: 'bar', field: 'FRL_pct', title: 'FRL % by Zone', color: CHART_COLORS.primary, unit: '%', max: 100 },
    'seat_disparity': { type: 'bar', field: 'seat_disparity', title: 'Seat Disparity by Zone', color: CHART_COLORS.quaternary, unit: '' },
    'avg_closest_zone_school_distance': { type: 'bar', field: 'avg_closest_school_distance', title: 'Avg Distance to Closest School', color: CHART_COLORS.secondary, unit: 'miles' },
    'avg_schools_in_attendance_area': { type: 'bar', field: 'schools_in_attendance_area', title: 'Schools in Attendance Area', color: CHART_COLORS.primary, unit: 'Count' },
    'boundary_cost': { type: 'none' },
    'avg_total_programs_per_zone': { type: 'bar', field: 'total_programs', title: 'Total Programs by Zone', color: CHART_COLORS.primary, unit: 'Count' },
    'avg_GE_per_zone': { type: 'bar', field: 'GE_programs', title: 'General Education by Zone', color: CHART_COLORS.secondary, unit: 'Count' },
    'avg_language_immersion_per_zone': { type: 'bar', field: 'language_immersion_count', title: 'Language Immersion by Zone', color: CHART_COLORS.tertiary, unit: 'Count' },
    'avg_special_ed_per_zone': { type: 'bar', field: 'special_ed_count', title: 'Special Ed by Zone', color: CHART_COLORS.quaternary, unit: 'Count' },
    'avg_math_score': { type: 'bar', field: 'avg_math_score', title: 'Math Scores by Zone', color: CHART_COLORS.primary, unit: 'Score' },
    'avg_eng_score': { type: 'bar', field: 'avg_eng_score', title: 'English Scores by Zone', color: CHART_COLORS.tertiary, unit: 'Score' },
};

const HISTORY_CATEGORIES = {
    'Div': [
        { key: 'theil_index', short: 'Div' },
        { key: 'FRL', short: 'FRL' },
        { key: 'seat_disparity', short: 'Seat' },
    ],
    'Dist': [
        { key: 'avg_closest_zone_school_distance', short: 'Dist' },
        { key: 'avg_schools_in_attendance_area', short: 'Sch' },
        { key: 'boundary_cost', short: 'Bnd' },
    ],
    'Prog': [
        { key: 'avg_total_programs_per_zone', short: 'Prg' },
        { key: 'avg_GE_per_zone', short: 'GE' },
        { key: 'avg_language_immersion_per_zone', short: 'LI' },
        { key: 'avg_special_ed_per_zone', short: 'SE' },
    ],
    'Perf': [
        { key: 'avg_math_score', short: 'Math' },
        { key: 'avg_eng_score', short: 'Eng' },
    ],
};

// ============================================================================
// Init
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

async function initAdmin() {
    initMap();
    setupResizeHandle();
    setupEventListeners();
    await loadSolutionSpace();
    applyFilters();
}

// ============================================================================
// Map (adapted from app.js)
// ============================================================================

function initMap() {
    map = L.map('map', { center: [37.76, -122.44], zoom: 12, zoomControl: true });
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap, &copy; CARTO',
        maxZoom: 19,
    }).addTo(map);
    loadSchoolLocations();
}

async function loadSchoolLocations() {
    try {
        const res = await fetch(`${API_BASE}/api/schools`);
        if (!res.ok) return;
        const data = await res.json();
        renderSchoolMarkers(data.schools);
    } catch (e) { console.error('Error loading schools:', e); }
}

function renderSchoolMarkers(schools) {
    if (!schools || !schools.length) return;
    if (schoolMarkersLayer) map.removeLayer(schoolMarkersLayer);

    schoolMarkersLayer = L.layerGroup();
    schools.forEach(s => {
        const icon = L.divIcon({
            html: '<div class="school-marker">&#127979;</div>',
            className: 'school-marker-container',
            iconSize: [20, 20], iconAnchor: [10, 10],
        });
        const marker = L.marker([s.lat, s.lon], { icon });
        let tip = `<strong>${s.name}</strong>`;
        if (s.total_capacity) tip += `<br>Capacity: ${s.total_capacity}`;
        marker.bindTooltip(tip, { direction: 'top', offset: [0, -10], className: 'school-tooltip' });
        marker.addTo(schoolMarkersLayer);
    });
    if (schoolsVisible) schoolMarkersLayer.addTo(map);
}

async function loadGeojson() {
    if (geojsonData) return geojsonData;
    const res = await fetch(`${API_BASE}/api/geojson`);
    if (!res.ok) throw new Error('Failed to load GeoJSON');
    geojsonData = await res.json();
    return geojsonData;
}

function showMapLoading(show) {
    document.getElementById('map-loading-overlay').classList.toggle('hidden', !show);
}

async function loadSolution(path) {
    showMapLoading(true);
    try {
        const [geojson, solRes] = await Promise.all([
            loadGeojson(),
            fetch(`${API_BASE}/api/solution/${encodeURIComponent(path)}`),
        ]);
        if (!solRes.ok) throw new Error('Failed to load solution');
        currentSolution = await solRes.json();
        renderMap(geojson);
        renderLegend();
        updateComparisonTable();
        refreshSingleChart();
        document.getElementById('map-placeholder').classList.add('hidden');
    } catch (e) {
        console.error('Error loading solution:', e);
    } finally {
        showMapLoading(false);
    }
}

function renderMap(geojson) {
    if (geojsonLayer) map.removeLayer(geojsonLayer);
    const { zones, zone_data, colors, zone_index_map } = currentSolution;

    geojsonLayer = L.geoJSON(geojson, {
        style: feature => {
            const bgId = String(feature.properties.BlockGroup);
            const zoneId = zones[bgId];
            const color = zoneId !== undefined ? (colors[String(zoneId)] || '#808080') : '#cccccc';
            return { fillColor: color, fillOpacity: 0.6, color: '#333', weight: 0.5 };
        },
        onEachFeature: (feature, layer) => {
            const bgId = String(feature.properties.BlockGroup);
            const zoneId = zones[bgId];
            const zd = zoneId !== undefined ? zone_data[String(zoneId)] : null;
            const zi = zoneId !== undefined && zone_index_map ? zone_index_map[String(zoneId)] : null;

            let tip = `<strong>BlockGroup: ${bgId}</strong>`;
            if (zi != null) tip += `<br><span>Zone ${zi}</span>`;
            if (zd) {
                tip += `<br>Students: ${Math.round(zd.ge_students)}`;
                tip += `<br>FRL: ${(zd.FRL_pct || 0).toFixed(1)}%`;
            }
            layer.bindTooltip(tip, { sticky: true });
            layer.on({
                mouseover: e => e.target.setStyle({ weight: 2, fillOpacity: 0.8 }),
                mouseout: e => geojsonLayer.resetStyle(e.target),
            });
        }
    }).addTo(map);

    map.invalidateSize();
    map.fitBounds(geojsonLayer.getBounds());
}

function renderLegend() {
    const el = document.getElementById('zone-legend');
    if (!el || !currentSolution) return;
    const { colors, zone_index_map } = currentSolution;
    if (!colors || !zone_index_map) { el.classList.add('hidden'); return; }

    const entries = Object.entries(zone_index_map)
        .map(([zoneId, idx]) => ({ zoneId, index: idx, color: colors[zoneId] || '#808080' }))
        .sort((a, b) => a.index - b.index);

    let html = '<div class="legend-header"><h4>Zones</h4>';
    if (schoolMarkersLayer) {
        const txt = schoolsVisible ? 'Hide Schools' : 'Show Schools';
        const cls = schoolsVisible ? 'active' : '';
        html += `<button id="toggle-schools-btn" class="toggle-schools-btn ${cls}">${txt}</button>`;
    }
    html += '</div>';
    entries.forEach(e => {
        html += `<div class="legend-item"><div class="legend-color" style="background:${e.color}"></div><span>${e.index}</span></div>`;
    });
    el.innerHTML = html;
    el.classList.remove('hidden');

    const btn = document.getElementById('toggle-schools-btn');
    if (btn) btn.addEventListener('click', toggleSchools);
}

function toggleSchools() {
    if (!schoolMarkersLayer || !map) return;
    schoolsVisible = !schoolsVisible;
    if (schoolsVisible) map.addLayer(schoolMarkersLayer);
    else map.removeLayer(schoolMarkersLayer);
    renderLegend();
}

// ============================================================================
// Comparison Table
// ============================================================================

function getPercentileRanking(p) {
    if (p >= 80) return 'excellent';
    if (p >= 60) return 'good';
    if (p >= 40) return 'average';
    if (p >= 20) return 'below-avg';
    return 'poor';
}

function formatValue(value, key) {
    if (value == null) return '-';
    if (key.includes('distance')) return value.toFixed(2) + ' mi';
    if (key === 'boundary_cost') return Math.round(value).toString();
    return value.toFixed(3);
}

function updateComparisonTable() {
    const container = document.getElementById('comparison-table-container');
    const ranks = currentSolution && currentSolution.percentile_ranks;
    if (!currentSolution || !currentSolution.metrics || !ranks) {
        container.innerHTML = '<p class="no-solution-msg">Adjust filters to see a solution</p>';
        return;
    }

    const metrics = currentSolution.metrics;
    const cats = {
        'Diversity': [
            { key: 'theil_index', name: 'Ethnic Diversity Index' },
            { key: 'FRL', name: 'FRL Representation' },
            { key: 'seat_disparity', name: 'Student Seat Imbalance' },
        ],
        'Distance': [
            { key: 'avg_closest_zone_school_distance', name: 'Avg Distance' },
            { key: 'avg_schools_in_attendance_area', name: 'Schools in Area' },
            { key: 'boundary_cost', name: 'Boundary Cost' },
        ],
        'Programs': [
            { key: 'avg_total_programs_per_zone', name: 'Total Programs' },
            { key: 'avg_GE_per_zone', name: 'General Education' },
            { key: 'avg_language_immersion_per_zone', name: 'Language Immersion' },
            { key: 'avg_special_ed_per_zone', name: 'Special Ed' },
        ],
        'Performance': [
            { key: 'avg_math_score', name: 'Math Scores' },
            { key: 'avg_eng_score', name: 'English Scores' },
        ],
    };

    let html = '<div class="category-list">';
    for (const [cat, ml] of Object.entries(cats)) {
        const catId = cat.toLowerCase();
        let pSum = 0, pCount = 0;
        ml.forEach(({ key }) => {
            const r = ranks[key];
            if (r && r.percentile != null) { pSum += r.percentile; pCount++; }
        });
        const avgP = pCount > 0 ? Math.round(pSum / pCount) : null;
        const avgR = avgP != null ? getPercentileRanking(avgP) : 'average';
        const avgBadge = avgP != null ? `<span class="percentile-indicator ${avgR}">${avgP}%</span>` : '';

        html += `<div class="category-card collapsed" data-category="${catId}" data-avg-badge="${encodeURIComponent(avgBadge)}">`;
        html += `<div class="category-card-header">`;
        html += `<span class="category-card-title"><span class="chevron">&#9654;</span> ${cat}</span>`;
        html += `<span class="category-avg-rank">${avgBadge}</span></div>`;
        html += `<table class="comparison-table category-metrics hidden">`;
        ml.forEach(({ key, name }) => {
            const v = metrics[key];
            const r = ranks[key];
            if (v === undefined || !r) return;
            const clickable = METRIC_CHART_CONFIG[key] && METRIC_CHART_CONFIG[key].type !== 'none';
            const sel = key === selectedMetricKey ? ' selected' : '';
            html += `<tr class="metric-row${sel}" data-key="${key}"${clickable ? ' data-clickable="true"' : ''}>`;
            html += `<td class="metric-name">${name}</td>`;
            html += `<td class="metric-rank"><span class="percentile-indicator ${r.ranking}">${r.percentile}%</span></td></tr>`;
        });
        html += `</table></div>`;
    }
    html += '</div>';
    container.innerHTML = html;

    container.querySelectorAll('.category-card').forEach(card => {
        card.querySelector('.category-card-header').addEventListener('click', () => {
            const collapsed = card.classList.contains('collapsed');
            if (collapsed) {
                card.classList.replace('collapsed', 'expanded');
                card.querySelector('.chevron').innerHTML = '&#9660;';
                card.querySelector('.category-avg-rank').innerHTML = '';
                card.querySelector('.category-metrics').classList.remove('hidden');
            } else {
                card.classList.replace('expanded', 'collapsed');
                card.querySelector('.chevron').innerHTML = '&#9654;';
                card.querySelector('.category-avg-rank').innerHTML = decodeURIComponent(card.dataset.avgBadge);
                card.querySelector('.category-metrics').classList.add('hidden');
            }
        });
    });

    container.querySelectorAll('.metric-row[data-clickable="true"]').forEach(row => {
        row.addEventListener('click', () => showSingleChart(row.dataset.key));
    });
}

// ============================================================================
// Charts
// ============================================================================

function showSingleChart(metricKey) {
    if (!currentSolution || !currentSolution.zone_data) return;
    const config = METRIC_CHART_CONFIG[metricKey];
    if (!config || config.type === 'none') return;

    const zoneData = currentSolution.zone_data;
    const zoneIndexMap = currentSolution.zone_index_map || {};
    const zoneIds = Object.keys(zoneData).sort((a, b) => Number(a) - Number(b));
    const labels = zoneIds.map(id => `Zone ${zoneIndexMap[id] || id}`);

    const canvas = document.getElementById('chart-single');
    document.getElementById('charts-panel').classList.remove('hidden');
    if (singleChart) { singleChart.destroy(); singleChart = null; }

    const defaultOpts = {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true } }
    };

    if (config.type === 'ethnicity') {
        const eKeys = ['Ethnicity_Black_or_African_American', 'Ethnicity_Hispanic/Latinx', 'Ethnicity_White', 'Ethnicity_Asian'];
        const eData = eKeys.map(k => zoneIds.map(id => ((zoneData[id].ethnicity_pcts || {})[k] || 0) * 100));
        const other = zoneIds.map((_, i) => Math.max(0, 100 - eData.reduce((s, a) => s + a[i], 0)));
        singleChart = new Chart(canvas.getContext('2d'), {
            type: 'bar',
            data: { labels, datasets: [
                { label: 'Black/African American', data: eData[0], backgroundColor: CHART_COLORS.ethnicities['Black/African American'] },
                { label: 'Hispanic/Latinx', data: eData[1], backgroundColor: CHART_COLORS.ethnicities['Hispanic/Latinx'] },
                { label: 'White', data: eData[2], backgroundColor: CHART_COLORS.ethnicities['White'] },
                { label: 'Asian', data: eData[3], backgroundColor: CHART_COLORS.ethnicities['Asian'] },
                { label: 'Other', data: other, backgroundColor: CHART_COLORS.ethnicities['Other'] },
            ]},
            options: { ...defaultOpts,
                plugins: { legend: { display: true, position: 'bottom', labels: { boxWidth: 12, font: { size: 10 } } } },
                scales: { x: { stacked: true }, y: { stacked: true, max: 100, title: { display: true, text: '%' } } }
            }
        });
    } else {
        const data = zoneIds.map(id => zoneData[id][config.field] || 0);
        const scaleOpts = { beginAtZero: true };
        if (config.max) scaleOpts.max = config.max;
        if (config.unit) scaleOpts.title = { display: true, text: config.unit };
        singleChart = new Chart(canvas.getContext('2d'), {
            type: 'bar',
            data: { labels, datasets: [{ label: config.title, data, backgroundColor: config.color || CHART_COLORS.primary }] },
            options: { ...defaultOpts, scales: { y: scaleOpts } }
        });
    }

    document.getElementById('charts-header').textContent = config.title;
    const sub = document.getElementById('charts-subtitle');
    if (sub) {
        const mv = currentSolution.metrics[metricKey];
        sub.textContent = mv != null ? `District-wide: ${formatValue(mv, metricKey)}` : '';
    }

    document.querySelectorAll('.metric-row.selected').forEach(r => r.classList.remove('selected'));
    const sel = document.querySelector(`.metric-row[data-key="${metricKey}"]`);
    if (sel) sel.classList.add('selected');
    selectedMetricKey = metricKey;
}

function refreshSingleChart() {
    if (selectedMetricKey) showSingleChart(selectedMetricKey);
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

    // Initialize filter bounds (all unconstrained)
    filterBounds = {};
    for (const col of Object.keys(solutionSpaceStats)) {
        filterBounds[col] = { min_bound: null, max_bound: null };
    }

    // Show core metrics by default
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
        if (b.min_bound != null || b.max_bound != null) {
            activeBounds[col] = b;
        }
    }

    const res = await adminFetch(`${API_BASE}/api/admin/filter`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bounds: activeBounds }),
    });
    const data = await res.json();

    updateSolutionBadge(data.solution_count, data.total_pareto);
    feasibleStats = data.feasible_stats || {};
    updateFeasibleBands();

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
                if (stat.direction === 'minimize') {
                    filterBounds[col].max_bound = newBound;
                } else {
                    filterBounds[col].min_bound = newBound;
                }
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
    if (!metrics) {
        panel.classList.add('hidden');
        return;
    }
    panel.classList.remove('hidden');

    // Show core metrics
    let html = '';
    for (const [col, stat] of Object.entries(solutionSpaceStats)) {
        if (!stat.is_core || !(col in metrics)) continue;
        html += `<div class="centroid-metric">
            <span class="centroid-metric-name">${stat.display_name}</span>
            <span class="centroid-metric-value">${formatMetricVal(col, metrics[col])}</span>
        </div>`;
    }
    container.innerHTML = html;
}

// ============================================================================
// Filter Sliders Rendering
// ============================================================================

function renderFilterSliders() {
    const container = document.getElementById('filter-sliders');
    container.innerHTML = '';

    // Group visible metrics by category
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
                <span class="filter-category-title"><span class="chevron">&#9660;</span> ${catLabel}</span>
                <span class="filter-category-count">${cols.length} metrics</span>
            </div>
            <div class="filter-category-metrics" data-cat="${cat}"></div>
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

    // Determine initial slider positions (full range if unconstrained)
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

    // Wire up sliders
    const minSlider = row.querySelector('.slider-min');
    const maxSlider = row.querySelector('.slider-max');

    const onSliderChange = () => {
        let loVal = parseFloat(minSlider.value);
        let hiVal = parseFloat(maxSlider.value);

        // Ensure min <= max
        if (loVal > hiVal) {
            loVal = hiVal;
            minSlider.value = loVal;
        }

        // Update filter bounds
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

    // Action buttons
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

    // Active band position
    const activeBand = row.querySelector('.slider-active-band');
    const leftPct = ((lo - gMin) / range) * 100;
    const widthPct = ((hi - lo) / range) * 100;
    activeBand.style.left = leftPct + '%';
    activeBand.style.width = widthPct + '%';

    // Feasible band
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

    // Value labels
    const b = filterBounds[col] || {};
    const loLabel = row.querySelector('.sv-lo');
    const hiLabel = row.querySelector('.sv-hi');
    loLabel.textContent = formatMetricVal(col, lo);
    hiLabel.textContent = formatMetricVal(col, hi);
    loLabel.classList.toggle('active', b.min_bound != null);
    hiLabel.classList.toggle('active', b.max_bound != null);

    // Show/hide loosen button based on whether filter is active
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

function updateFeasibleBands() {
    for (const col of visibleMetrics) {
        const row = document.querySelector(`.metric-slider-row[data-col="${col}"]`);
        if (row) updateSliderVisuals(row, col);
    }
}

function loosenMetric(col) {
    const stat = solutionSpaceStats[col];
    const b = filterBounds[col];
    if (!b) return;

    const expansionFactor = 0.3;
    if (stat.direction === 'minimize' && b.max_bound != null) {
        const globalMax = stat.max;
        b.max_bound = Math.min(globalMax, b.max_bound + (globalMax - b.max_bound) * expansionFactor);
        if (Math.abs(b.max_bound - globalMax) < getStep(stat.min, globalMax) * 0.5) b.max_bound = null;
    } else if (stat.direction === 'maximize' && b.min_bound != null) {
        const globalMin = stat.min;
        b.min_bound = Math.max(globalMin, b.min_bound - (b.min_bound - globalMin) * expansionFactor);
        if (Math.abs(b.min_bound - globalMin) < getStep(globalMin, stat.max) * 0.5) b.min_bound = null;
    }

    updateSliderUI(col);
    applyFilters();
}

// ============================================================================
// Metric Search (add non-core metrics)
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
// Solution History
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
    const ranks = currentSolution.percentile_ranks || {};
    const categoryScores = {};
    for (const [cat, metrics] of Object.entries(HISTORY_CATEGORIES)) {
        categoryScores[cat] = getCategoryPercentile(ranks, metrics);
    }

    savedSolutions.push({
        index, path,
        solutionData: JSON.parse(JSON.stringify(currentSolution)),
        label: `Solution #${index}`,
        pros: '', cons: '',
        timestamp: new Date().toISOString(),
        categoryScores,
    });
    currentViewedIndex = index;
    renderSolutionHistory();
}

function getCategoryPercentile(ranks, categoryMetrics) {
    if (!ranks) return null;
    let sum = 0, count = 0;
    for (const m of categoryMetrics) {
        const r = ranks[m.key];
        if (r && r.percentile != null) { sum += r.percentile; count++; }
    }
    return count > 0 ? Math.round(sum / count) : null;
}

function viewSavedSolution(index) {
    const entry = savedSolutions.find(s => s.index === index);
    if (!entry) return;
    currentViewedIndex = index;
    currentSolution = entry.solutionData;
    loadGeojson().then(geojson => {
        renderMap(geojson);
        renderLegend();
        updateComparisonTable();
        refreshSingleChart();
        document.getElementById('map-placeholder').classList.add('hidden');
    });
    renderSolutionHistory();
    updateProsConsPanel();
}

function toggleHistoryExpanded() {
    const container = document.getElementById('solution-history');
    const toggleBtn = document.getElementById('solution-history-toggle');
    if (!container) return;
    historyExpanded = !historyExpanded;
    container.classList.toggle('collapsed', !historyExpanded);
    container.classList.toggle('expanded', historyExpanded);
    if (toggleBtn) {
        toggleBtn.innerHTML = historyExpanded ? '&#9660;' : '&#9650;';
        toggleBtn.title = historyExpanded ? 'Collapse' : 'Expand';
    }
    updateProsConsPanel();
}

function renderSolutionHistory() {
    const container = document.getElementById('solution-history');
    const cardsContainer = document.getElementById('solution-cards');
    if (!container || !cardsContainer) return;

    if (savedSolutions.length === 0) {
        container.classList.add('hidden');
        return;
    }

    container.classList.remove('hidden');
    if (!container.classList.contains('expanded') && !container.classList.contains('collapsed')) {
        container.classList.add('collapsed');
    }

    cardsContainer.innerHTML = '';
    savedSolutions.forEach(entry => {
        const card = document.createElement('div');
        card.className = 'solution-card' + (entry.index === currentViewedIndex ? ' active' : '');

        const top = document.createElement('div');
        top.className = 'solution-card-top';
        top.innerHTML = `<span class="solution-card-number">${entry.index}</span><span class="solution-card-label">${entry.label}</span>`;
        card.appendChild(top);

        const metricsRow = document.createElement('div');
        metricsRow.className = 'solution-card-metrics';
        const scores = entry.categoryScores || {};
        for (const [cat, pct] of Object.entries(scores)) {
            if (pct === null) continue;
            const ranking = getPercentileRanking(pct);
            const badge = document.createElement('span');
            badge.className = `solution-metric-badge percentile-indicator ${ranking}`;
            badge.innerHTML = `${cat} ${pct}%`;
            metricsRow.appendChild(badge);
        }
        card.appendChild(metricsRow);
        card.addEventListener('click', () => viewSavedSolution(entry.index));
        cardsContainer.appendChild(card);
    });

    const toggleBtn = document.getElementById('solution-history-toggle');
    if (toggleBtn && !toggleBtn._wired) {
        toggleBtn.addEventListener('click', toggleHistoryExpanded);
        toggleBtn._wired = true;
    }
}

function updateProsConsPanel() {
    const panel = document.getElementById('solution-proscons-panel');
    const prosEl = document.getElementById('solution-pros-textarea');
    const consEl = document.getElementById('solution-cons-textarea');
    const targetLabel = document.getElementById('solution-proscons-target');
    if (!panel || !prosEl || !consEl) return;

    const entry = savedSolutions.find(s => s.index === currentViewedIndex);
    if (!historyExpanded || !entry) {
        panel.classList.add('hidden');
        return;
    }
    panel.classList.remove('hidden');
    if (targetLabel) targetLabel.textContent = `Solution #${entry.index}`;

    [['solution-pros-textarea', 'pros'], ['solution-cons-textarea', 'cons']].forEach(([id, field]) => {
        const el = document.getElementById(id);
        const fresh = el.cloneNode(true);
        el.parentNode.replaceChild(fresh, el);
        fresh.id = id;
        fresh.value = entry[field] || '';
        fresh.addEventListener('blur', () => {
            const text = fresh.value.trim();
            if (entry[field] !== text) { entry[field] = text; renderSolutionHistory(); }
        });
    });
}

// ============================================================================
// Event Listeners & Resize Handle
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

    const chartsClose = document.getElementById('charts-close');
    if (chartsClose) {
        chartsClose.addEventListener('click', () => {
            document.getElementById('charts-panel').classList.add('hidden');
            if (singleChart) { singleChart.destroy(); singleChart = null; }
            selectedMetricKey = null;
            document.querySelectorAll('.metric-row.selected').forEach(r => r.classList.remove('selected'));
        });
    }
}

function setupResizeHandle() {
    const handle = document.getElementById('resize-handle-right');
    const main = document.querySelector('main');
    if (!handle || !main) return;

    let startX, startWidth;
    handle.addEventListener('mousedown', e => {
        e.preventDefault();
        startX = e.clientX;
        startWidth = document.getElementById('filter-panel').offsetWidth;
        handle.classList.add('dragging');

        const onMove = ev => {
            const diff = startX - ev.clientX;
            const newWidth = Math.max(300, Math.min(700, startWidth + diff));
            main.style.gridTemplateColumns = `1fr 4px ${newWidth}px`;
            map && map.invalidateSize();
        };
        const onUp = () => {
            handle.classList.remove('dragging');
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
        };
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
    });
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
