// SFUSD Zoning Dashboard - Shared Module
// Common state, map, charts, comparison table, solution history, school markers,
// legend, formatting, and metrics config used by both user and admin pages.

const API_BASE = '';

// Populated from /api/metrics-config on startup
let metricsConfig = null;

// Map state
let map = null;
let geojsonLayer = null;
let geojsonData = null;
let schoolMarkersLayer = null;
let schoolsVisible = false;
let currentSolution = null;

// Chart state
let singleChart = null;
let selectedMetricKey = null;

// Solution history state
let savedSolutions = [];
let currentViewedIndex = null;
let historyExpanded = false;
const MAX_SAVED_SOLUTIONS = 30;

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

// Category display constants
const CATEGORY_SHORT = { diversity: 'Div', distance: 'Dist', programs: 'Prog', quality: 'Perf', structure: 'Struct' };
const CATEGORY_DISPLAY = { diversity: 'Diversity', distance: 'Distance', programs: 'Programs', quality: 'Performance', structure: 'Structure' };
// Reverse lookup: display name → category key (e.g. "Performance" → "quality")
const DISPLAY_TO_CATEGORY = Object.fromEntries(
    Object.entries(CATEGORY_DISPLAY).map(([k, v]) => [v, k])
);

// Page-specific hooks (set by app.js or admin.js before calling shared init)
let pageHooks = {
    rightPanelSelector: '#chat-panel',
    onSolutionLoaded: () => {},
    onSolutionLoadError: (err) => console.error('Solution load error:', err),
    buildCardExtras: (card, entry) => {},
    trackEvent: (name, props) => {},
};

// ============================================================================
// Metrics Config
// ============================================================================

async function fetchMetricsConfig() {
    try {
        const res = await fetch(`${API_BASE}/api/metrics-config`);
        if (res.ok) metricsConfig = await res.json();
    } catch (e) {
        console.warn('Failed to fetch metrics config:', e);
    }
}

function getChartConfig() {
    if (!metricsConfig) return {};
    const config = {};
    for (const m of metricsConfig.metrics) {
        config[m.column] = m.chart || { type: 'none' };
    }
    return config;
}

function getCoreCategories() {
    if (!metricsConfig) return {};
    const result = {};
    for (const [catKey, catDisplay] of Object.entries(CATEGORY_DISPLAY)) {
        const metrics = metricsConfig.metrics
            .filter(m => m.category === catKey && m.is_core)
            .map(m => ({ key: m.column, name: m.short_name || m.display_name }));
        if (metrics.length > 0) result[catDisplay] = metrics;
    }
    return result;
}


function getPercentileRanking(percentile) {
    if (percentile >= 80) return 'excellent';
    if (percentile >= 60) return 'good';
    if (percentile >= 40) return 'average';
    if (percentile >= 20) return 'below-avg';
    return 'poor';
}

function formatValue(value, key) {
    if (value === undefined || value === null) return '-';
    if (key.includes('distance')) return value.toFixed(2) + ' miles';
    if (key.includes('rating')) return value.toFixed(1);
    if (key === 'boundary_cost') return Math.round(value).toString();
    if (key === 'compactness') return value.toFixed(1);
    if (key.includes('index') || key.includes('Index')) return value.toFixed(2);
    return value.toFixed(2);
}

// ============================================================================
// Map
// ============================================================================

function initMap() {
    map = L.map('map', {
        center: [37.76, -122.44],
        zoom: 12,
        zoomControl: true,
    });

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
    } catch (e) {
        console.error('Error loading school locations:', e);
    }
}

function renderSchoolMarkers(schools) {
    if (!schools || schools.length === 0) return;
    if (schoolMarkersLayer) map.removeLayer(schoolMarkersLayer);

    schoolMarkersLayer = L.layerGroup();
    schools.forEach(school => {
        const icon = L.divIcon({
            html: '<div class="school-marker">&#127979;</div>',
            className: 'school-marker-container',
            iconSize: [20, 20],
            iconAnchor: [10, 10],
        });
        const marker = L.marker([school.lat, school.lon], { icon });

        const geCapacity = school.programs && school.programs['GE'] !== undefined
            ? school.programs['GE'] : school.total_capacity;
        let tip = `<div class="school-tooltip-content"><strong>${school.name}</strong>`;
        if (geCapacity !== undefined) tip += `<br>GE Seats: ${geCapacity}`;
        tip += `</div>`;

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

        pageHooks.trackEvent('solution_loaded', {
            solution_path: path,
            num_zones: Object.keys(currentSolution.zones || {}).length
                ? new Set(Object.values(currentSolution.zones)).size : 0,
            status: currentSolution.status,
        });

        pageHooks.onSolutionLoaded(path);
    } catch (error) {
        console.error('Error loading solution:', error);
        pageHooks.onSolutionLoadError(error);
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
            layer.bindTooltip(createTooltip(bgId, zi, zd), { sticky: true });
            layer.on({
                mouseover: e => e.target.setStyle({ weight: 2, fillOpacity: 0.8 }),
                mouseout: e => geojsonLayer.resetStyle(e.target),
            });
        }
    }).addTo(map);

    map.invalidateSize();
    map.fitBounds(geojsonLayer.getBounds());
}

function createTooltip(bgId, zoneIndex, demographics) {
    let content = `<strong>BlockGroup: ${bgId}</strong>`;
    if (zoneIndex !== undefined && zoneIndex !== null) {
        content += `<br><span class="zone-info">Zone ${zoneIndex}</span>`;
    }
    if (demographics) {
        content += `<br>Students: ${Math.round(demographics.ge_students)}`;
        content += `<br>FRL: ${(demographics.FRL_pct || 0).toFixed(1)}%`;
    }
    return content;
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
        html += `<button id="toggle-schools-btn" class="toggle-schools-btn ${cls}" title="Toggle school markers">${txt}</button>`;
    }
    html += '</div>';
    for (const entry of entries) {
        html += `<div class="legend-item">
            <div class="legend-color" style="background-color: ${entry.color}"></div>
            <span class="legend-label">Zone ${entry.index}</span>
        </div>`;
    }
    el.innerHTML = html;
    el.classList.remove('hidden');

    const btn = document.getElementById('toggle-schools-btn');
    if (btn) btn.addEventListener('click', toggleSchoolMarkers);
}

function toggleSchoolMarkers() {
    if (!schoolMarkersLayer || !map) return;
    schoolsVisible = !schoolsVisible;
    if (schoolsVisible) map.addLayer(schoolMarkersLayer);
    else map.removeLayer(schoolMarkersLayer);

    const btn = document.getElementById('toggle-schools-btn');
    if (btn) {
        btn.textContent = schoolsVisible ? 'Hide Schools' : 'Show Schools';
        btn.classList.toggle('active', schoolsVisible);
    }
    pageHooks.trackEvent('schools_toggled', { visible: schoolsVisible });
}

// ============================================================================
// Charts
// ============================================================================

function showSingleChart(metricKey) {
    if (!currentSolution || !currentSolution.zone_data) return;
    const chartConfigs = getChartConfig();
    const config = chartConfigs[metricKey];
    if (!config || config.type === 'none') return;

    const zoneData = currentSolution.zone_data;
    const zoneIndexMap = currentSolution.zone_index_map || {};
    const zoneIds = Object.keys(zoneData).sort((a, b) => Number(a) - Number(b));
    const labels = zoneIds.map(id => `Zone ${zoneIndexMap[id] || id}`);

    const canvas = document.getElementById('chart-single');
    document.getElementById('charts-panel').classList.remove('hidden');
    if (singleChart) { singleChart.destroy(); singleChart = null; }

    const defaultOpts = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true } }
    };

    if (config.type === 'ethnicity') {
        const ethnicityDisplay = metricsConfig ? metricsConfig.ethnicities.display : [
            { key: 'Ethnicity_Black_or_African_American', label: 'Black/African American' },
            { key: 'Ethnicity_Hispanic/Latinx', label: 'Hispanic/Latinx' },
            { key: 'Ethnicity_White', label: 'White' },
            { key: 'Ethnicity_Asian', label: 'Asian' },
        ];
        const eKeys = ethnicityDisplay.map(e => e.key);
        const eLabels = ethnicityDisplay.map(e => e.label);
        const eData = eKeys.map(k =>
            zoneIds.map(id => ((zoneData[id].ethnicity_pcts || {})[k] || 0) * 100)
        );
        const other = zoneIds.map((_, i) => Math.max(0, 100 - eData.reduce((s, a) => s + a[i], 0)));
        const datasets = eLabels.map((label, i) => ({
            label,
            data: eData[i],
            backgroundColor: CHART_COLORS.ethnicities[label] || CHART_COLORS.primary,
        }));
        datasets.push({ label: 'Other', data: other, backgroundColor: CHART_COLORS.ethnicities['Other'] });

        singleChart = new Chart(canvas.getContext('2d'), {
            type: 'bar',
            data: { labels, datasets },
            options: {
                ...defaultOpts,
                plugins: { legend: { display: true, position: 'bottom', labels: { boxWidth: 12, font: { size: 10 } } } },
                scales: { x: { stacked: true }, y: { stacked: true, max: 100, title: { display: true, text: '%' } } }
            }
        });
    } else {
        const data = zoneIds.map(id => {
            const val = zoneData[id] ? zoneData[id][config.field] : undefined;
            return (val === undefined || val === null) ? null : val;
        });

        const chartContainer = document.getElementById('single-chart-container');
        let noDataMsg = chartContainer.querySelector('.no-zone-data-msg');

        if (data.every(v => v === null)) {
            canvas.style.display = 'none';
            if (!noDataMsg) {
                noDataMsg = document.createElement('p');
                noDataMsg.className = 'no-zone-data-msg';
                chartContainer.appendChild(noDataMsg);
            }
            noDataMsg.textContent = 'Zone-level breakdown not available for this metric.';
            noDataMsg.style.display = 'block';
        } else {
            canvas.style.display = '';
            if (noDataMsg) noDataMsg.style.display = 'none';

            const scaleOpts = { beginAtZero: true };
            if (config.max) scaleOpts.max = config.max;
            if (config.unit) scaleOpts.title = { display: true, text: config.unit };

            singleChart = new Chart(canvas.getContext('2d'), {
                type: 'bar',
                data: { labels, datasets: [{ label: config.title, data, backgroundColor: config.color || CHART_COLORS.primary }] },
                options: { ...defaultOpts, scales: { y: scaleOpts } }
            });
        }
    }

    document.getElementById('charts-header').textContent = config.title;
    const subtitle = document.getElementById('charts-subtitle');
    if (subtitle) {
        const mv = currentSolution.metrics[metricKey];
        const displayValue = mv !== undefined ? formatValue(mv, metricKey) : null;
        subtitle.textContent = displayValue ? `District-wide value: ${displayValue}` : '';
    }

    document.querySelectorAll('.metric-row.selected').forEach(r => r.classList.remove('selected'));
    const sel = document.querySelector(`.metric-row[data-key="${metricKey}"]`);
    if (sel) sel.classList.add('selected');
    selectedMetricKey = metricKey;

    pageHooks.trackEvent('metric_chart_clicked', { metric_key: metricKey });
}

function refreshSingleChart() {
    if (selectedMetricKey && currentSolution && currentSolution.zone_data) {
        showSingleChart(selectedMetricKey);
    }
}

// ============================================================================
// Comparison Table
// ============================================================================

function updateComparisonTable() {
    const container = document.getElementById('comparison-table-container');
    const ranks = currentSolution && currentSolution.percentile_ranks;

    if (!currentSolution || !currentSolution.metrics || !ranks) {
        container.innerHTML = '<p class="no-solution-msg">Select a solution to see comparison</p>';
        return;
    }

    const metrics = currentSolution.metrics;
    const categories = getCoreCategories();
    const catPercentiles = currentSolution.category_percentiles || {};

    let html = '<div class="category-list">';
    for (const [category, metricList] of Object.entries(categories)) {
        const catKey = DISPLAY_TO_CATEGORY[category];
        const catShort = CATEGORY_SHORT[catKey];
        const avgP = catShort && catPercentiles[catShort] != null
            ? Math.round(catPercentiles[catShort])
            : null;
        const avgR = avgP !== null ? getPercentileRanking(avgP) : 'average';
        const avgBadge = avgP !== null
            ? `<span class="percentile-indicator ${avgR}">${avgP}%</span>` : '';

        html += `<div class="category-card collapsed" data-category="${catKey}" data-avg-badge="${encodeURIComponent(avgBadge)}">`;
        html += `<div class="category-card-header">`;
        html += `<span class="category-card-title"><span class="chevron">&#9654;</span> ${category}</span>`;
        html += `<span class="category-avg-rank">${avgBadge}</span>`;
        html += `</div>`;

        html += `<table class="comparison-table category-metrics hidden">`;
        for (const { key, name } of metricList) {
            const value = metrics[key];
            const rank = ranks[key];
            if (value === undefined || value === null) continue;

            const mConfig = metricsConfig && metricsConfig.metrics.find(m => m.column === key);
            const isInfoOnly = mConfig && mConfig.direction == null;

            const rawValue = rank && rank.raw_value !== undefined
                ? formatValue(rank.raw_value, key)
                : formatValue(value, key);
            const rankBadge = isInfoOnly
                ? ''
                : (rank
                    ? `<span class="percentile-indicator ${rank.ranking}">${rank.percentile}%</span>`
                    : `<span class="percentile-indicator">-</span>`);
            const chartConfigs = getChartConfig();
            const clickable = chartConfigs[key] && chartConfigs[key].type !== 'none';
            const selectedClass = key === selectedMetricKey ? ' selected' : '';

            html += `<tr class="metric-row${selectedClass}" data-key="${key}"${clickable ? ' data-clickable="true"' : ''}>`;
            html += `<td class="metric-name">${name}</td>`;
            html += `<td class="metric-value">${rawValue}</td>`;
            html += `<td class="metric-rank">${rankBadge}</td>`;
            html += `</tr>`;
        }
        html += `</table></div>`;
    }
    html += '</div>';
    container.innerHTML = html;

    container.querySelectorAll('.category-card').forEach(card => {
        card.querySelector('.category-card-header').addEventListener('click', () => {
            const isCollapsed = card.classList.contains('collapsed');
            if (isCollapsed) {
                card.classList.remove('collapsed');
                card.classList.add('expanded');
                card.querySelector('.chevron').innerHTML = '&#9660;';
                card.querySelector('.category-avg-rank').innerHTML = '';
                card.querySelector('.category-metrics').classList.remove('hidden');
            } else {
                card.classList.remove('expanded');
                card.classList.add('collapsed');
                card.querySelector('.chevron').innerHTML = '&#9654;';
                card.querySelector('.category-avg-rank').innerHTML = decodeURIComponent(card.dataset.avgBadge);
                card.querySelector('.category-metrics').classList.add('hidden');
            }
        });
    });

    container.querySelectorAll('.metric-row[data-clickable="true"]').forEach(row => {
        row.addEventListener('click', () => showSingleChart(row.dataset.key));
    });

    pageHooks.trackEvent('comparison_table_viewed', {
        num_metrics: Object.keys(metrics).length,
        solution_path: currentSolution.path,
    });
}

// ============================================================================
// Solution History
// ============================================================================

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

    pageHooks.trackEvent('solution_revisited', {
        solution_index: index,
        label: entry.label,
    });
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
        toggleBtn.title = historyExpanded ? 'Collapse panel' : 'Expand panel';
    }

    updateProsConsPanel();
    if (map) setTimeout(() => map.invalidateSize(), 300);
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
        card.className = 'solution-card' +
            (entry.index === currentViewedIndex ? ' active' : '') +
            ((entry.pros || entry.cons) ? ' has-note' : '');
        card.dataset.index = entry.index;

        const top = document.createElement('div');
        top.className = 'solution-card-top';

        const number = document.createElement('span');
        number.className = 'solution-card-number';
        number.textContent = entry.index;

        const label = document.createElement('span');
        label.className = 'solution-card-label';
        label.textContent = entry.label;
        label.title = entry.label;

        top.appendChild(number);
        top.appendChild(label);

        // Let page-specific code add extras (e.g. note button)
        pageHooks.buildCardExtras(top, entry);

        card.appendChild(top);

        const metricsRow = document.createElement('div');
        metricsRow.className = 'solution-card-metrics';
        const scores = entry.categoryScores || {};
        for (const [cat, pct] of Object.entries(scores)) {
            if (pct === null) continue;
            const badge = document.createElement('span');
            const ranking = getPercentileRanking(pct);
            badge.className = `solution-metric-badge ${ranking}`;
            badge.innerHTML = `<span class="metric-badge-label">${cat}</span> ${pct}%`;
            metricsRow.appendChild(badge);
        }
        card.appendChild(metricsRow);

        card.addEventListener('click', () => viewSavedSolution(entry.index));
        cardsContainer.appendChild(card);
    });

    const activeCard = cardsContainer.querySelector('.solution-card.active');
    if (activeCard) {
        activeCard.scrollIntoView({ behavior: 'smooth', inline: 'nearest', block: 'nearest' });
    }

    const toggleBtn = document.getElementById('solution-history-toggle');
    if (toggleBtn && !toggleBtn._wired) {
        toggleBtn.addEventListener('click', toggleHistoryExpanded);
        toggleBtn._wired = true;
    }
}

function updateProsConsPanel() {
    const panel = document.getElementById('solution-proscons-panel');
    const prosTextarea = document.getElementById('solution-pros-textarea');
    const consTextarea = document.getElementById('solution-cons-textarea');
    const targetLabel = document.getElementById('solution-proscons-target');
    if (!panel || !prosTextarea || !consTextarea) return;

    const entry = savedSolutions.find(s => s.index === currentViewedIndex);

    if (!historyExpanded || !entry) {
        panel.classList.add('hidden');
        return;
    }

    panel.classList.remove('hidden');
    if (targetLabel) targetLabel.textContent = `Solution #${entry.index}`;

    for (const [id, field] of [['solution-pros-textarea', 'pros'], ['solution-cons-textarea', 'cons']]) {
        const el = document.getElementById(id);
        const fresh = el.cloneNode(true);
        el.parentNode.replaceChild(fresh, el);
        fresh.id = id;
        fresh.value = entry[field] || '';
        fresh.addEventListener('blur', () => {
            const text = fresh.value.trim();
            if (entry[field] !== text) {
                entry[field] = text;
                renderSolutionHistory();
                pageHooks.trackEvent('solution_proscons_updated', { solution_index: entry.index, field, length: text.length });
            }
        });
        fresh.addEventListener('keydown', (e) => { if (e.key === 'Escape') fresh.blur(); });
    }
}

// ============================================================================
// Shared UI Setup
// ============================================================================

function setupResizeHandle() {
    const handle = document.getElementById('resize-handle-right');
    const main = document.querySelector('main');
    if (!handle || !main) return;

    let isResizing = false;
    let startX = 0;
    let startY = 0;
    let startPanelWidth = 0;

    handle.addEventListener('mousedown', e => {
        isResizing = true;
        startX = e.clientX;
        startY = e.clientY;
        const panel = document.querySelector(pageHooks.rightPanelSelector);
        startPanelWidth = panel ? panel.offsetWidth : 350;
        handle.classList.add('resizing');

        const isMobileView = window.innerWidth <= 900;
        document.body.style.cursor = isMobileView ? 'row-resize' : 'col-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();
    });

    document.addEventListener('mousemove', e => {
        if (!isResizing) return;
        const isMobileView = window.innerWidth <= 900;

        if (isMobileView) {
            const deltaY = startY - e.clientY;
            const newHeight = Math.max(200, Math.min(600, startPanelWidth + deltaY));
            main.style.gridTemplateRows = `1fr 4px ${newHeight}px`;
        } else {
            const deltaX = e.clientX - startX;
            const newWidth = Math.max(250, Math.min(700, startPanelWidth - deltaX));
            main.style.gridTemplateColumns = `1fr 4px ${newWidth}px`;
        }

        if (map) map.invalidateSize();
    });

    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            handle.classList.remove('resizing');
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        }
    });
}

function setupChartsClose() {
    const chartsClose = document.getElementById('charts-close');
    if (chartsClose) {
        chartsClose.addEventListener('click', () => {
            document.getElementById('charts-panel').classList.add('hidden');
            document.querySelectorAll('.metric-row.selected').forEach(r => r.classList.remove('selected'));
            selectedMetricKey = null;
            if (singleChart) { singleChart.destroy(); singleChart = null; }
        });
    }
}
