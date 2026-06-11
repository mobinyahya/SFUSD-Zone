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
let schoolsControl = null;
let highwaysLayer = null;
let highwaysVisible = true;
let highwaysControl = null;
let currentSolution = null;
let currentSolutionPath = null;

// SES (FRL) overlay state
let sesOverlayActive = false;
let blockgroupFrl = null;          // { bgId: frlPct(0-100) }
let zoneBoundariesLayer = null;
let zoneBoundariesGeneration = 0;
let sesOverlayControl = null;

// Sequential color ramp for FRL % (low → high). ColorBrewer OrRd-style.
const FRL_RAMP = [
    { max: 20,  color: '#fff7ec', label: '< 20%' },
    { max: 40,  color: '#fdd49e', label: '20–40%' },
    { max: 60,  color: '#fdbb84', label: '40–60%' },
    { max: 75,  color: '#fc8d59', label: '60–75%' },
    { max: 90,  color: '#e34a33', label: '75–90%' },
    { max: 101, color: '#b30000', label: '≥ 90%' },
];

function frlColor(pct) {
    if (pct == null || isNaN(pct)) return '#dddddd';
    for (const bin of FRL_RAMP) {
        if (pct < bin.max) return bin.color;
    }
    return FRL_RAMP[FRL_RAMP.length - 1].color;
}

// AALPI (racial) overlay state
let aalpiOverlayActive = false;
let blockgroupAalpi = null;        // { bgId: aalpiPct(0-100) }
let aalpiOverlayControl = null;

// Sequential color ramp for AALPI % (low → high). ColorBrewer Purples-style.
const AALPI_RAMP = [
    { max: 20,  color: '#fcfbfd', label: '< 20%' },
    { max: 40,  color: '#dadaeb', label: '20–40%' },
    { max: 60,  color: '#bcbddc', label: '40–60%' },
    { max: 75,  color: '#9e9ac8', label: '60–75%' },
    { max: 90,  color: '#756bb1', label: '75–90%' },
    { max: 101, color: '#54278f', label: '≥ 90%' },
];

function aalpiColor(pct) {
    if (pct == null || isNaN(pct)) return '#dddddd';
    for (const bin of AALPI_RAMP) {
        if (pct < bin.max) return bin.color;
    }
    return AALPI_RAMP[AALPI_RAMP.length - 1].color;
}

// Chart state
let singleChart = null;
let selectedMetricKey = null;
const normalizeOverrides = {};

// Version state
let versions = [];
let currentVersionId = null;
const MAX_VERSIONS = 30;

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
const CATEGORY_SHORT = { diversity: 'Div', proximity: 'Prox', programs: 'Prog', quality: 'Perf', structure: 'Struct' };
const CATEGORY_DISPLAY = { diversity: 'Diversity', proximity: 'Proximity', programs: 'Availability', quality: 'Performance', structure: 'Structure' };
// Reverse lookup: display name → category key (e.g. "Performance" → "quality")
const DISPLAY_TO_CATEGORY = Object.fromEntries(
    Object.entries(CATEGORY_DISPLAY).map(([k, v]) => [v, k])
);

// Page-specific hooks (set by app.js or admin.js before calling shared init)
let pageHooks = {
    rightPanelSelector: '#chat-panel',
    onSolutionLoaded: () => { },
    onSolutionLoadError: (err) => console.error('Solution load error:', err),
    buildCardExtras: (card, entry) => { },
    trackEvent: (name, props) => { },
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
    if (key.endsWith('_mad') || key === 'seat_disparity') return (value * 100).toFixed(1);
    if (key.includes('distance')) return value.toFixed(3);
    if (key.includes('rating')) return value.toFixed(2);
    if (key === 'boundary_cost') return Math.round(value).toString();
    if (key === 'compactness') return value.toFixed(1);
    if (key.includes('index') || key.includes('Index')) return value.toFixed(2);
    return value.toFixed(3);
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

    L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap, &copy; CARTO',
        maxZoom: 19,
    }).addTo(map);

    map.createPane('highwaysPane');
    map.getPane('highwaysPane').style.zIndex = 675;
    map.getPane('highwaysPane').style.pointerEvents = 'none';

    map.getPane('tooltipPane').style.zIndex = 690;

    addSesOverlayControl();
    addAalpiOverlayControl();
    addHighwaysToggleControl();
    addSchoolsToggleControl();
    loadSchoolLocations();
    loadHighwaysOverlay();
}

function addSchoolsToggleControl() {
    if (schoolsControl) return;
    const Control = L.Control.extend({
        options: { position: 'topright' },
        onAdd: function () {
            const div = L.DomUtil.create('div', 'schools-toggle-control');
            div.innerHTML = renderSchoolsToggleHtml();
            L.DomEvent.disableClickPropagation(div);
            L.DomEvent.disableScrollPropagation(div);
            div.querySelector('.schools-toggle-btn').addEventListener('click', toggleSchoolMarkers);
            return div;
        }
    });
    schoolsControl = new Control();
    schoolsControl.addTo(map);
}

function renderSchoolsToggleHtml() {
    const active = schoolsVisible ? 'active' : '';
    const label = schoolsVisible ? 'Hide schools' : 'Show schools';
    return `
        <button class="schools-toggle-btn ${active}" title="Toggle school markers">
            <span class="schools-toggle-dot">&#127979;</span>${label}
        </button>
    `;
}

async function loadHighwaysOverlay() {
    try {
        const res = await fetch('/static/sf_highways.geojson');
        if (!res.ok) return;
        const data = await res.json();
        highwaysLayer = L.geoJSON(data, {
            pane: 'highwaysPane',
            interactive: false,
            style: {
                color: '#555',
                weight: 2.5,
                opacity: 0.55,
                dashArray: '2 6',
                lineCap: 'round',
            },
        });
        if (highwaysVisible) highwaysLayer.addTo(map);
    } catch (e) {
        console.error('Error loading highways overlay:', e);
    }
}

function addHighwaysToggleControl() {
    if (highwaysControl) return;
    const Control = L.Control.extend({
        options: { position: 'topright' },
        onAdd: function () {
            const div = L.DomUtil.create('div', 'highways-toggle-control');
            div.innerHTML = renderHighwaysToggleHtml();
            L.DomEvent.disableClickPropagation(div);
            L.DomEvent.disableScrollPropagation(div);
            div.querySelector('.highways-toggle-btn').addEventListener('click', toggleHighways);
            return div;
        }
    });
    highwaysControl = new Control();
    highwaysControl.addTo(map);
}

function renderHighwaysToggleHtml() {
    const active = highwaysVisible ? 'active' : '';
    const label = highwaysVisible ? 'Hide highways' : 'Show highways';
    return `
        <button class="highways-toggle-btn ${active}" title="Toggle highway overlay">
            <span class="highways-toggle-dot"></span>${label}
        </button>
    `;
}

function toggleHighways() {
    highwaysVisible = !highwaysVisible;
    if (highwaysLayer) {
        if (highwaysVisible) highwaysLayer.addTo(map);
        else map.removeLayer(highwaysLayer);
    }
    const root = highwaysControl && highwaysControl.getContainer();
    if (root) {
        root.innerHTML = renderHighwaysToggleHtml();
        root.querySelector('.highways-toggle-btn').addEventListener('click', toggleHighways);
    }
    pageHooks.trackEvent('highways_toggled', { visible: highwaysVisible });
}

async function loadSchoolLocations() {
    try {
        const res = await fetch(`${API_BASE}/api/schools`);
        if (!res.ok) return;
        const data = await res.json();
        const zoneSchools = data.schools.filter(s => s.category !== 'Citywide');
        renderSchoolMarkers(zoneSchools);
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

        const tip = `<div class="school-tooltip-content"><strong>${school.name}</strong></div>`;

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

function buildActivityQuery() {
    const params = new URLSearchParams();
    const pid = localStorage.getItem('participant_id') || '';
    if (pid) params.set('participant_id', pid);
    if (typeof sessionId !== 'undefined' && sessionId) params.set('session_id', sessionId);
    const qs = params.toString();
    return qs ? `?${qs}` : '';
}

async function loadSolution(path) {
    showMapLoading(true);
    try {
        const [geojson, solRes] = await Promise.all([
            loadGeojson(),
            fetch(`${API_BASE}/api/solution/${encodeURIComponent(path)}${buildActivityQuery()}`),
        ]);
        if (!solRes.ok) throw new Error('Failed to load solution');
        currentSolution = await solRes.json();
        currentSolutionPath = path;

        renderMap(geojson);
        refreshZoneBoundaries();
        loadBlockgroupFrl();
        loadBlockgroupAalpi();
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

async function loadSolutionByCode(code) {
    const res = await fetch(`${API_BASE}/api/solution-by-code/${encodeURIComponent(code)}${buildActivityQuery()}`);
    if (!res.ok) return null;
    const { path } = await res.json();
    await loadSolution(path);
    return path;
}

function blockgroupStyle(feature) {
    const { zones, colors } = currentSolution;
    const bgId = String(feature.properties.BlockGroup);
    const zoneId = zones[bgId];

    if (sesOverlayActive) {
        const pct = blockgroupFrl ? blockgroupFrl[bgId] : null;
        return {
            fillColor: frlColor(pct),
            fillOpacity: zoneId !== undefined ? 0.85 : 0.25,
            color: '#999',
            weight: 0.3,
        };
    }

    if (aalpiOverlayActive) {
        const pct = blockgroupAalpi ? blockgroupAalpi[bgId] : null;
        return {
            fillColor: aalpiColor(pct),
            fillOpacity: zoneId !== undefined ? 0.85 : 0.25,
            color: '#999',
            weight: 0.3,
        };
    }
    const color = zoneId !== undefined ? (colors[String(zoneId)] || '#808080') : '#cccccc';
    return { fillColor: color, fillOpacity: 0.6, color: '#333', weight: 0.5 };
}

function renderMap(geojson) {
    if (geojsonLayer) map.removeLayer(geojsonLayer);
    const { zones, zone_data, zone_index_map } = currentSolution;

    geojsonLayer = L.geoJSON(geojson, {
        style: blockgroupStyle,
        onEachFeature: (feature, layer) => {
            const bgId = String(feature.properties.BlockGroup);
            const zoneId = zones[bgId];
            layer._zoneId = zoneId;
            const zd = zoneId !== undefined ? zone_data[String(zoneId)] : null;
            const zi = zoneId !== undefined && zone_index_map ? zone_index_map[String(zoneId)] : null;
            layer.bindTooltip(() => createTooltip(bgId, zi, zd), { sticky: true });
            layer.on({
                mouseover: () => {
                    if (zoneId === undefined) return;
                    geojsonLayer.eachLayer(l => {
                        if (l._zoneId === zoneId) {
                            const base = blockgroupStyle(l.feature);
                            l.setStyle({ weight: Math.max(2, base.weight), fillOpacity: Math.min(1, base.fillOpacity + 0.15) });
                        }
                    });
                },
                mouseout: () => {
                    if (zoneId === undefined) return;
                    geojsonLayer.eachLayer(l => {
                        if (l._zoneId === zoneId) geojsonLayer.resetStyle(l);
                    });
                },
            });
        }
    }).addTo(map);

    map.invalidateSize();
    map.fitBounds(geojsonLayer.getBounds());
}

// ============================================================================
// SES Overlay (FRL choropleth + bold zone borders)
// ============================================================================

function addSesOverlayControl() {
    if (sesOverlayControl) return;
    const Control = L.Control.extend({
        options: { position: 'topright' },
        onAdd: function () {
            const div = L.DomUtil.create('div', 'ses-overlay-control');
            div.innerHTML = renderSesControlHtml();
            L.DomEvent.disableClickPropagation(div);
            L.DomEvent.disableScrollPropagation(div);
            div.querySelector('.ses-toggle-btn').addEventListener('click', toggleSesOverlay);
            return div;
        }
    });
    sesOverlayControl = new Control();
    sesOverlayControl.addTo(map);
}

function renderSesControlHtml() {
    const active = sesOverlayActive ? 'active' : '';
    const label = sesOverlayActive ? 'Hide SES overlay' : 'Show SES overlay';
    let legend = '';
    if (sesOverlayActive) {
        legend = '<div class="ses-legend"><div class="ses-legend-title">FRL %</div>';
        for (const bin of FRL_RAMP) {
            legend += `<div class="ses-legend-row">
                <span class="ses-legend-swatch" style="background:${bin.color}"></span>
                <span class="ses-legend-label">${bin.label}</span>
            </div>`;
        }
        legend += '</div>';
    }
    return `
        <button class="ses-toggle-btn ${active}" title="Toggle Free/Reduced Lunch overlay">
            <span class="ses-toggle-dot"></span>${label}
        </button>
        ${legend}
    `;
}

function refreshSesOverlayControl() {
    const root = sesOverlayControl && sesOverlayControl.getContainer();
    if (!root) return;
    root.innerHTML = renderSesControlHtml();
    root.querySelector('.ses-toggle-btn').addEventListener('click', toggleSesOverlay);
}

function toggleSesOverlay() {
    sesOverlayActive = !sesOverlayActive;

    if (sesOverlayActive && aalpiOverlayActive) {
        aalpiOverlayActive = false;
        refreshAalpiOverlayControl();
    }

    refreshSesOverlayControl();

    if (geojsonLayer) {
        geojsonLayer.setStyle(blockgroupStyle);
    }

    if (zoneBoundariesLayer) {
        if (sesOverlayActive) zoneBoundariesLayer.addTo(map);
        else map.removeLayer(zoneBoundariesLayer);
    }

    if (sesOverlayActive && !blockgroupFrl) {
        loadBlockgroupFrl().then(() => {
            if (sesOverlayActive && geojsonLayer) geojsonLayer.setStyle(blockgroupStyle);
        });
    }

    pageHooks.trackEvent('ses_overlay_toggled', { visible: sesOverlayActive });
}

async function loadBlockgroupFrl() {
    if (blockgroupFrl) return blockgroupFrl;
    try {
        const res = await fetch(`${API_BASE}/api/blockgroup-frl`);
        if (!res.ok) throw new Error('Failed to load FRL data');
        const data = await res.json();
        blockgroupFrl = data.frl_pct || {};
    } catch (e) {
        console.error(e);
    }
    return blockgroupFrl;
}

// ============================================================================
// AALPI (Racial) Overlay (choropleth + bold zone borders)
// ============================================================================

function addAalpiOverlayControl() {
    if (aalpiOverlayControl) return;
    const Control = L.Control.extend({
        options: { position: 'topright' },
        onAdd: function () {
            const div = L.DomUtil.create('div', 'aalpi-overlay-control');
            div.innerHTML = renderAalpiControlHtml();
            L.DomEvent.disableClickPropagation(div);
            L.DomEvent.disableScrollPropagation(div);
            div.querySelector('.aalpi-toggle-btn').addEventListener('click', toggleAalpiOverlay);
            return div;
        }
    });
    aalpiOverlayControl = new Control();
    aalpiOverlayControl.addTo(map);
}

function renderAalpiControlHtml() {
    const active = aalpiOverlayActive ? 'active' : '';
    const label = aalpiOverlayActive ? 'Hide racial overlay' : 'Show racial overlay';
    let legend = '';
    if (aalpiOverlayActive) {
        legend = '<div class="aalpi-legend"><div class="aalpi-legend-title">AALPI %</div>';
        for (const bin of AALPI_RAMP) {
            legend += `<div class="aalpi-legend-row">
                <span class="aalpi-legend-swatch" style="background:${bin.color}"></span>
                <span class="aalpi-legend-label">${bin.label}</span>
            </div>`;
        }
        legend += '</div>';
    }
    return `
        <button class="aalpi-toggle-btn ${active}" title="Toggle AALPI (Black + Hispanic/Latinx + Pacific Islander) overlay">
            <span class="aalpi-toggle-dot"></span>${label}
        </button>
        ${legend}
    `;
}

function refreshAalpiOverlayControl() {
    const root = aalpiOverlayControl && aalpiOverlayControl.getContainer();
    if (!root) return;
    root.innerHTML = renderAalpiControlHtml();
    root.querySelector('.aalpi-toggle-btn').addEventListener('click', toggleAalpiOverlay);
}

function toggleAalpiOverlay() {
    aalpiOverlayActive = !aalpiOverlayActive;

    if (aalpiOverlayActive && sesOverlayActive) {
        sesOverlayActive = false;
        refreshSesOverlayControl();
    }

    refreshAalpiOverlayControl();

    if (geojsonLayer) {
        geojsonLayer.setStyle(blockgroupStyle);
    }

    if (zoneBoundariesLayer) {
        if (aalpiOverlayActive || sesOverlayActive) zoneBoundariesLayer.addTo(map);
        else map.removeLayer(zoneBoundariesLayer);
    }

    if (aalpiOverlayActive && !blockgroupAalpi) {
        loadBlockgroupAalpi().then(() => {
            if (aalpiOverlayActive && geojsonLayer) geojsonLayer.setStyle(blockgroupStyle);
        });
    }

    pageHooks.trackEvent('aalpi_overlay_toggled', { visible: aalpiOverlayActive });
}

async function loadBlockgroupAalpi() {
    if (blockgroupAalpi) return blockgroupAalpi;
    try {
        const res = await fetch(`${API_BASE}/api/blockgroup-aalpi`);
        if (!res.ok) throw new Error('Failed to load AALPI data');
        const data = await res.json();
        blockgroupAalpi = data.aalpi_pct || {};
    } catch (e) {
        console.error(e);
    }
    return blockgroupAalpi;
}

async function refreshZoneBoundaries() {
    const gen = ++zoneBoundariesGeneration;
    if (zoneBoundariesLayer) {
        map.removeLayer(zoneBoundariesLayer);
        zoneBoundariesLayer = null;
    }
    if (!currentSolutionPath) return;
    try {
        const res = await fetch(`${API_BASE}/api/zone-boundaries/${encodeURIComponent(currentSolutionPath)}`);
        if (gen !== zoneBoundariesGeneration) return;
        if (!res.ok) return;
        const fc = await res.json();
        if (gen !== zoneBoundariesGeneration) return;
        if (zoneBoundariesLayer) {
            map.removeLayer(zoneBoundariesLayer);
            zoneBoundariesLayer = null;
        }
        zoneBoundariesLayer = L.geoJSON(fc, {
            interactive: false,
            style: () => ({
                color: '#1a1a1a',
                weight: 2.5,
                opacity: 0.95,
                fillOpacity: 0,
            }),
        });
        if (sesOverlayActive) zoneBoundariesLayer.addTo(map);
    } catch (e) {
        console.error('Failed to load zone boundaries:', e);
    }
}

function formatEthnicityName(key) {
    return key.replace('Ethnicity_', '').replace(/_/g, ' ').replace('/', '/');
}

function createTooltip(bgId, zoneIndex, demographics) {
    const bgFrl = (sesOverlayActive && blockgroupFrl) ? blockgroupFrl[bgId] : null;
    const bgFrlRow = bgFrl != null
        ? `<div class="tooltip-row tooltip-bg-frl"><span>This blockgroup FRL</span><span>${bgFrl.toFixed(1)}%</span></div>`
        : '';

    const bgAalpi = (aalpiOverlayActive && blockgroupAalpi) ? blockgroupAalpi[bgId] : null;
    const bgAalpiRow = bgAalpi != null
        ? `<div class="tooltip-row tooltip-bg-aalpi"><span>This blockgroup AALPI</span><span>${bgAalpi.toFixed(1)}%</span></div>`
        : '';

    const overlayRow = bgFrlRow || bgAalpiRow;

    if (!demographics) {
        return `<strong>BlockGroup: ${bgId}</strong><br>Unassigned${overlayRow}`;
    }

    const d = demographics;
    const pct = v => v != null ? (v * 100).toFixed(1) + '%' : 'N/A';
    const num = (v, dec = 0) => v != null ? v.toFixed(dec) : 'N/A';

    let html = `<div class="tooltip-header">Zone ${zoneIndex ?? '?'}</div>`;
    if (overlayRow) html += `<div class="tooltip-section">${overlayRow}</div>`;

    // Demographics
    html += `<div class="tooltip-section"><div class="tooltip-section-label">Demographics</div>`;
    html += `<div class="tooltip-row"><span>Students</span><span>${Math.round(d.ge_students)}</span></div>`;
    if (d.ethnicity_pcts) {
        for (const [key, val] of Object.entries(d.ethnicity_pcts)) {
            html += `<div class="tooltip-row tooltip-sub"><span>${formatEthnicityName(key)}</span><span>${pct(val)}</span></div>`;
        }
    }
    html += `<div class="tooltip-row"><span>FRL</span><span>${num(d.FRL_pct, 1)}%</span></div>`;
    if (d.seat_disparity != null) {
        html += `<div class="tooltip-row"><span>Seat Disparity</span><span>${pct(d.seat_disparity)}</span></div>`;
    }
    html += `</div>`;

    // Quality
    html += `<div class="tooltip-section"><div class="tooltip-section-label">Quality</div>`;
    if (d.avg_math_score != null) {
        html += `<div class="tooltip-row"><span>Avg Math</span><span>${num(d.avg_math_score, 2)}</span></div>`;
    }
    if (d.avg_eng_score != null) {
        html += `<div class="tooltip-row"><span>Avg English</span><span>${num(d.avg_eng_score, 2)}</span></div>`;
    }
    html += `</div>`;

    // Access
    html += `<div class="tooltip-section"><div class="tooltip-section-label">Access</div>`;
    if (d.avg_any_ge_school_distance != null) {
        html += `<div class="tooltip-row"><span>Avg GE School Dist</span><span>${num(d.avg_any_ge_school_distance, 2)} mi</span></div>`;
    }
    if (d.ge_schools_within_half_mile != null) {
        html += `<div class="tooltip-row"><span>GE Schools &lt;0.5 mi</span><span>${num(d.ge_schools_within_half_mile, 2)}</span></div>`;
    }
    html += `</div>`;

    return html;
}

function renderLegend() {
    const el = document.getElementById('zone-legend');
    if (!el || !currentSolution) return;
    const { colors, zone_index_map } = currentSolution;
    if (!colors || !zone_index_map) { el.classList.add('hidden'); return; }

    const entries = Object.entries(zone_index_map)
        .map(([zoneId, idx]) => ({ zoneId, index: idx, color: colors[zoneId] || '#808080' }))
        .sort((a, b) => a.index - b.index);

    let html = '<h4>Zones</h4>';
    for (const entry of entries) {
        html += `<div class="legend-item">
            <div class="legend-color" style="background-color: ${entry.color}"></div>
            <span class="legend-label">Zone ${entry.index}</span>
        </div>`;
    }
    el.innerHTML = html;
    el.classList.remove('hidden');
}

function toggleSchoolMarkers() {
    if (!schoolMarkersLayer || !map) return;
    schoolsVisible = !schoolsVisible;
    if (schoolsVisible) map.addLayer(schoolMarkersLayer);
    else map.removeLayer(schoolMarkersLayer);

    const root = schoolsControl && schoolsControl.getContainer();
    if (root) {
        root.innerHTML = renderSchoolsToggleHtml();
        root.querySelector('.schools-toggle-btn').addEventListener('click', toggleSchoolMarkers);
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

    // Activate side-by-side layout: selected card on left, chart on right
    document.getElementById('comparison-panel').classList.add('chart-active');

    // Hide non-selected cards, expand the selected one
    document.querySelectorAll('.category-card').forEach(card => {
        const containsMetric = card.querySelector(`.metric-row[data-key="${metricKey}"]`);
        if (!containsMetric) {
            card.classList.add('chart-hidden');
        } else {
            card.classList.remove('chart-hidden', 'collapsed');
            card.classList.add('expanded');
            card.querySelector('.chevron').innerHTML = '&#9660;';
            card.querySelector('.category-avg-rank').innerHTML = '';
            card.querySelector('.category-metrics').classList.remove('hidden');
        }
    });

    const zoneData = currentSolution.zone_data;
    const zoneIndexMap = currentSolution.zone_index_map || {};
    const zoneIds = Object.keys(zoneData).sort((a, b) => Number(a) - Number(b));
    const labels = zoneIds.map(id => `Zone ${zoneIndexMap[id] || id}`);

    const canvas = document.getElementById('chart-single');
    document.getElementById('inline-chart-area').classList.remove('hidden');
    if (singleChart) { singleChart.destroy(); singleChart = null; }

    const defaultOpts = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true } }
    };

    const isNormalized = metricKey in normalizeOverrides
        ? normalizeOverrides[metricKey]
        : !!config.normalize;

    const normBtn = document.getElementById('charts-normalize-toggle');
    if (normBtn) {
        normBtn.style.display = config.type === 'ethnicity' ? 'none' : '';
        normBtn.textContent = isNormalized ? 'Raw' : '%Dev';
        normBtn.classList.toggle('active', isNormalized);
    }

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
        let data = zoneIds.map(id => {
            const val = zoneData[id] ? zoneData[id][config.field] : undefined;
            if (val === undefined || val === null) return null;
            return config.unit === '%' ? val * 100 : val;
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

            let scaleOpts = { beginAtZero: true };
            if (isNormalized) {
                const valid = data.filter(v => v !== null);
                const mean = valid.reduce((s, v) => s + v, 0) / valid.length;
                if (mean !== 0) {
                    data = data.map(v => v === null ? null : ((v - mean) / mean) * 100);
                }
                scaleOpts = { title: { display: true, text: '% Deviation from Avg' } };
            } else {
                if (config.max) scaleOpts.max = config.max;
                if (config.unit) scaleOpts.title = { display: true, text: config.unit };
            }

            const zoneColors = zoneIds.map(id => (currentSolution.colors || {})[id] || CHART_COLORS.primary);
            singleChart = new Chart(canvas.getContext('2d'), {
                type: 'bar',
                data: { labels, datasets: [{ label: config.title, data, backgroundColor: zoneColors }] },
                options: { ...defaultOpts, scales: { y: scaleOpts } }
            });
        }
    }

    const mConfig = metricsConfig && metricsConfig.metrics.find(m => m.column === metricKey);
    document.getElementById('charts-header').textContent = (mConfig && mConfig.display_name) || config.title;

    const descEl = document.getElementById('charts-description');
    if (descEl) {
        descEl.textContent = (mConfig && mConfig.description) || '';
    }

    const subtitle = document.getElementById('charts-subtitle');
    if (subtitle) {
        const mv = currentSolution.metrics[metricKey];
        const displayValue = mv !== undefined ? formatValue(mv, metricKey) : null;
        const unit = mConfig && mConfig.chart && mConfig.chart.unit;
        subtitle.textContent = displayValue ? `Value for this map: ${displayValue}${unit ? ' ' + unit : ''}` : '';
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
// Version Management
// ============================================================================

function viewVersion(versionId) {
    const entry = versions.find(v => v.id === versionId);
    if (!entry || !entry.solutionData) return;

    currentVersionId = versionId;
    currentSolution = entry.solutionData;
    currentSolutionPath = entry.solutionPath || currentSolutionPath;

    loadGeojson().then(geojson => {
        renderMap(geojson);
        refreshZoneBoundaries();
        loadBlockgroupFrl();
        loadBlockgroupAalpi();
        renderLegend();
        updateComparisonTable();
        refreshSingleChart();
        document.getElementById('map-placeholder').classList.add('hidden');
    });

    pageHooks.trackEvent('version_switched', {
        version_id: versionId,
        label: entry.label,
    });
}

function getLatestVersionId() {
    if (versions.length === 0) return null;
    return versions[versions.length - 1].id;
}

function isLatestVersion(versionId) {
    return versionId === getLatestVersionId();
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
            document.getElementById('inline-chart-area').classList.add('hidden');
            document.querySelectorAll('.metric-row.selected').forEach(r => r.classList.remove('selected'));
            selectedMetricKey = null;
            if (singleChart) { singleChart.destroy(); singleChart = null; }

            // Remove side-by-side layout
            document.getElementById('comparison-panel').classList.remove('chart-active');

            // Restore all category cards to collapsed state
            document.querySelectorAll('.category-card').forEach(card => {
                card.classList.remove('expanded', 'chart-hidden');
                card.classList.add('collapsed');
                card.querySelector('.chevron').innerHTML = '&#9654;';
                card.querySelector('.category-avg-rank').innerHTML = decodeURIComponent(card.dataset.avgBadge);
                card.querySelector('.category-metrics').classList.add('hidden');
            });
        });
    }

    const normToggle = document.getElementById('charts-normalize-toggle');
    if (normToggle) {
        normToggle.addEventListener('click', () => {
            if (!selectedMetricKey) return;
            const chartConfigs = getChartConfig();
            const config = chartConfigs[selectedMetricKey];
            const current = selectedMetricKey in normalizeOverrides
                ? normalizeOverrides[selectedMetricKey]
                : !!config?.normalize;
            normalizeOverrides[selectedMetricKey] = !current;
            refreshSingleChart();
        });
    }
}
