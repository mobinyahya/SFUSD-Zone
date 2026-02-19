// SFUSD Zoning Dashboard - Main Application

const API_BASE = '';

// State
let map = null;
let geojsonLayer = null;
let geojsonData = null;
let schoolMarkersLayer = null;
let schoolsVisible = false;
let currentSolution = null;
let sessionId = null;
let isProcessing = false;
let posthogApiKey = null;

// Solution history state
let savedSolutions = [];
let currentViewedIndex = null;
const MAX_SAVED_SOLUTIONS = 30;

// Chart instances
let charts = {};
let singleChart = null;
let selectedMetricKey = null;

// DOM Elements
const mapPlaceholder = document.getElementById('map-placeholder');
const mapLoadingOverlay = document.getElementById('map-loading-overlay');
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const chatSend = document.getElementById('chat-send');
const chatInputArea = document.getElementById('chat-input-area');
const loadingOverlay = document.getElementById('loading-overlay');
const resizeHandleRight = document.getElementById('resize-handle-right');
const mainContainer = document.querySelector('main');

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

// Initialize
document.addEventListener('DOMContentLoaded', init);

async function init() {
    // Fetch PostHog config from backend
    try {
        const configResponse = await fetch(`${API_BASE}/api/config`);
        if (configResponse.ok) {
            const config = await configResponse.json();
            posthogApiKey = config.posthog_api_key;
        }
    } catch (error) {
        console.warn('Failed to fetch PostHog config:', error);
    }

    // Check localStorage for prior consent + participant ID
    const hasConsent = localStorage.getItem('posthog_consent') === 'true';
    const participantId = localStorage.getItem('participant_id');

    if (hasConsent && participantId) {
        // Returning user: initialize PostHog and proceed directly
        initPostHog(participantId);
        await initApp();
    } else {
        // New user: show consent banner
        showConsentBanner();
    }
}

function showConsentBanner() {
    const banner = document.getElementById('consent-banner');
    const acceptBtn = document.getElementById('consent-accept');
    banner.classList.remove('hidden');

    acceptBtn.addEventListener('click', () => {
        localStorage.setItem('posthog_consent', 'true');
        banner.classList.add('hidden');
        showIdentifyModal();
    });
}

function showIdentifyModal() {
    const modal = document.getElementById('identify-modal');
    const submitBtn = document.getElementById('identify-submit');
    const input = document.getElementById('participant-id');
    modal.classList.remove('hidden');

    const handleSubmit = async () => {
        const participantId = input.value.trim();
        if (!participantId) return;

        localStorage.setItem('participant_id', participantId);
        modal.classList.add('hidden');

        initPostHog(participantId);
        await initApp();
    };

    submitBtn.addEventListener('click', handleSubmit);
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSubmit();
    });
}

function initPostHog(participantId) {
    if (!posthogApiKey || typeof posthog === 'undefined') {
        console.warn('PostHog not available (no API key or SDK not loaded)');
        return;
    }

    posthog.init(posthogApiKey, {
        api_host: 'https://us.i.posthog.com',
        autocapture: true,
        capture_pageview: true,
        session_recording: {
            maskAllInputs: false,
        },
    });

    posthog.identify(participantId);
    posthog.capture('session_started', { participant_id: participantId });
    console.log('PostHog initialized for participant:', participantId);
}

async function initApp() {
    // Set up event listeners FIRST - these are critical and should work even if other things fail
    setupEventListeners();
    try {
        initMap();
    } catch (error) {
        console.error('Failed to initialize map:', error);
    }

    // Send initial message to trigger clustering
    try {
        await sendInitialMessage();
    } catch (error) {
        console.error('Failed to send initial message:', error);
        // Make sure processing state is reset if initial message fails
        setProcessing(false);
    }
}

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

    // Load school locations
    loadSchoolLocations();
}

async function loadSchoolLocations() {
    try {
        const response = await fetch(`${API_BASE}/api/schools`);
        if (!response.ok) throw new Error('Failed to load school locations');

        const data = await response.json();
        renderSchoolMarkers(data.schools);
    } catch (error) {
        console.error('Error loading school locations:', error);
    }
}

function renderSchoolMarkers(schools) {
    if (!schools || schools.length === 0) return;

    // Remove existing school markers if any
    if (schoolMarkersLayer) {
        map.removeLayer(schoolMarkersLayer);
    }

    // Create a layer group for school markers
    schoolMarkersLayer = L.layerGroup();

    schools.forEach(school => {
        // Create custom icon using divIcon with school building emoji
        const icon = L.divIcon({
            html: '<div class="school-marker">🏫</div>',
            className: 'school-marker-container',
            iconSize: [20, 20],
            iconAnchor: [10, 10],
        });

        // Create marker
        const marker = L.marker([school.lat, school.lon], { icon });

        let tooltipContent = `<div class="school-tooltip-content"><strong>${school.name}</strong>`;
        if (school.total_capacity !== undefined) {
            tooltipContent += `<br>Capacity: ${school.total_capacity}`;
        }
        if (school.programs && Object.keys(school.programs).length > 0) {
            const activePrograms = Object.entries(school.programs).filter(([p, c]) => c > 0);
            if (activePrograms.length > 0) {
                tooltipContent += `<div class="school-programs-list"><strong>Programs:</strong><ul>` + 
                    activePrograms.map(([p, c]) => `<li>${p}: ${c}</li>`).join('') + 
                    `</ul></div>`;
            }
        }
        tooltipContent += `</div>`;

        // Add tooltip with school name
        marker.bindTooltip(tooltipContent, {
            direction: 'top',
            offset: [0, -10],
            className: 'school-tooltip',
        });

        // Add to layer group
        marker.addTo(schoolMarkersLayer);
    });

    // Only add layer to map if schools should be visible
    if (schoolsVisible) {
        schoolMarkersLayer.addTo(map);
    }
}

// Metric-to-chart-data mapping
const METRIC_CHART_CONFIG = {
    'theil_index': { type: 'ethnicity', title: 'Ethnic Composition by Zone', color: null },
    'FRL': { type: 'bar', field: 'FRL_pct', title: 'FRL % by Zone', color: CHART_COLORS.primary, unit: '%', max: 100 },
    'seat_disparity': { type: 'bar', field: 'seat_disparity', title: 'Seat Disparity by Zone', color: CHART_COLORS.quaternary, unit: '' },
    'avg_closest_zone_school_distance': { type: 'bar', field: 'avg_closest_school_distance', title: 'Avg Distance to Closest School', color: CHART_COLORS.secondary, unit: 'miles' },
    'avg_schools_in_attendance_area': { type: 'bar', field: 'schools_in_attendance_area', title: 'Schools in Attendance Area', color: CHART_COLORS.primary, unit: 'Count' },
    'boundary_cost': { type: 'none' },
    'avg_total_programs_per_zone': { type: 'bar', field: 'total_programs', title: 'Total Programs by Zone', color: CHART_COLORS.primary, unit: 'Count' },
    'avg_GE_per_zone': { type: 'bar', field: 'GE_programs', title: 'General Education by Zone', color: CHART_COLORS.secondary, unit: 'Count' },
    'avg_language_immersion_per_zone': { type: 'bar', field: 'language_immersion_count', title: 'Language Immersion by Zone', color: CHART_COLORS.tertiary, unit: 'Count' },
    'avg_special_ed_per_zone': { type: 'bar', field: 'special_ed_count', title: 'Special Ed by Zone', color: CHART_COLORS.quaternary, unit: 'Count' },
    'avg_greatschools_rating': { type: 'bar', field: 'avg_greatschools_rating', title: 'GreatSchools Rating by Zone', color: CHART_COLORS.secondary, unit: 'Rating', max: 10 },
    'avg_math_score': { type: 'bar', field: 'avg_math_score', title: 'Math Scores by Zone', color: CHART_COLORS.primary, unit: 'Score' },
    'avg_eng_score': { type: 'bar', field: 'avg_eng_score', title: 'English Scores by Zone', color: CHART_COLORS.tertiary, unit: 'Score' },
    'avg_suspension_index': { type: 'bar', field: 'avg_suspension_index', title: 'Suspension Index by Zone', color: CHART_COLORS.quinary, unit: 'Index', max: 5 },
};

function showSingleChart(metricKey) {
    if (!currentSolution || !currentSolution.zone_data) return;

    const config = METRIC_CHART_CONFIG[metricKey];
    if (!config || config.type === 'none') return;

    const zoneData = currentSolution.zone_data;
    const zoneIndexMap = currentSolution.zone_index_map || {};
    const zoneIds = Object.keys(zoneData).sort((a, b) => Number(a) - Number(b));
    const labels = zoneIds.map(id => `Zone ${zoneIndexMap[id] || id}`);

    const canvas = document.getElementById('chart-single');
    const chartsPanel = document.getElementById('charts-panel');
    chartsPanel.classList.remove('hidden');

    // Destroy previous chart
    if (singleChart) {
        singleChart.destroy();
        singleChart = null;
    }

    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true } }
    };

    if (config.type === 'ethnicity') {
        // Stacked ethnicity bar chart
        const ethnicityKeys = ['Ethnicity_Black_or_African_American', 'Ethnicity_Hispanic/Latinx', 'Ethnicity_White', 'Ethnicity_Asian'];
        const ethnicityData = ethnicityKeys.map(key =>
            zoneIds.map(id => {
                const pcts = zoneData[id].ethnicity_pcts || {};
                return (pcts[key] || 0) * 100;
            })
        );
        const otherData = zoneIds.map((id, idx) => {
            const sum = ethnicityData.reduce((acc, arr) => acc + arr[idx], 0);
            return Math.max(0, 100 - sum);
        });

        singleChart = new Chart(canvas.getContext('2d'), {
            type: 'bar',
            data: {
                labels,
                datasets: [
                    { label: 'Black/African American', data: ethnicityData[0], backgroundColor: CHART_COLORS.ethnicities['Black/African American'] },
                    { label: 'Hispanic/Latinx', data: ethnicityData[1], backgroundColor: CHART_COLORS.ethnicities['Hispanic/Latinx'] },
                    { label: 'White', data: ethnicityData[2], backgroundColor: CHART_COLORS.ethnicities['White'] },
                    { label: 'Asian', data: ethnicityData[3], backgroundColor: CHART_COLORS.ethnicities['Asian'] },
                    { label: 'Other', data: otherData, backgroundColor: CHART_COLORS.ethnicities['Other'] },
                ]
            },
            options: {
                ...defaultOptions,
                plugins: { legend: { display: true, position: 'bottom', labels: { boxWidth: 12, font: { size: 10 } } } },
                scales: { x: { stacked: true }, y: { stacked: true, max: 100, title: { display: true, text: '%' } } }
            }
        });
    } else {
        // Simple bar chart
        const data = zoneIds.map(id => zoneData[id][config.field] || 0);
        const scaleOpts = { beginAtZero: true };
        if (config.max) scaleOpts.max = config.max;
        if (config.unit) scaleOpts.title = { display: true, text: config.unit };

        singleChart = new Chart(canvas.getContext('2d'), {
            type: 'bar',
            data: { labels, datasets: [{ label: config.title, data, backgroundColor: config.color }] },
            options: { ...defaultOptions, scales: { y: scaleOpts } }
        });
    }

    // Update header and subtitle with district-wide value
    document.getElementById('charts-header').textContent = config.title;
    const subtitle = document.getElementById('charts-subtitle');
    if (subtitle) {
        const metricValue = currentSolution.metrics[metricKey];
        const displayValue = metricValue !== undefined ? formatValue(metricValue, metricKey) : null;
        subtitle.textContent = displayValue
            ? `District-wide value: ${displayValue}`
            : '';
    }

    // Highlight selected metric row
    document.querySelectorAll('.metric-row.selected').forEach(r => r.classList.remove('selected'));
    const selectedRow = document.querySelector(`.metric-row[data-key="${metricKey}"]`);
    if (selectedRow) selectedRow.classList.add('selected');

    selectedMetricKey = metricKey;

    trackEvent('metric_chart_clicked', { metric_key: metricKey });
}

async function loadGeojson() {
    if (geojsonData) return geojsonData;

    const response = await fetch(`${API_BASE}/api/geojson`);
    if (!response.ok) throw new Error('Failed to load GeoJSON');

    geojsonData = await response.json();
    return geojsonData;
}

async function loadSolution(path) {
    console.log('[loadSolution] Loading solution from path:', path);
    showMapLoading(true);

    try {
        const [geojson, solutionResponse] = await Promise.all([
            loadGeojson(),
            fetch(`${API_BASE}/api/solution/${encodeURIComponent(path)}`)
        ]);

        console.log('[loadSolution] Response status:', solutionResponse.status);
        if (!solutionResponse.ok) throw new Error('Failed to load solution');

        currentSolution = await solutionResponse.json();
        console.log('[loadSolution] Solution loaded:', {
            zones: Object.keys(currentSolution.zones || {}).length,
            demographics: Object.keys(currentSolution.demographics || {}).length,
            zone_data: Object.keys(currentSolution.zone_data || {}).length,
            status: currentSolution.status
        });

        trackEvent('solution_loaded', {
            solution_path: path,
            num_zones: Object.keys(currentSolution.zones || {}).length ? new Set(Object.values(currentSolution.zones)).size : 0,
            status: currentSolution.status,
        });

        renderMap(geojson);
        renderLegend();
        updateComparisonTable();
        refreshSingleChart();

        mapPlaceholder.classList.add('hidden');
    } catch (error) {
        console.error('[loadSolution] Error loading solution:', error);
        addMessage('system', 'Failed to load solution. Please try again.');
    } finally {
        showMapLoading(false);
    }
}

function renderMap(geojson) {
    if (geojsonLayer) {
        map.removeLayer(geojsonLayer);
    }

    const { zones, zone_data, colors, zone_index_map } = currentSolution;

    geojsonLayer = L.geoJSON(geojson, {
        style: feature => {
            const bgId = String(feature.properties.BlockGroup);
            const zoneId = zones[bgId];
            const color = zoneId !== undefined ? (colors[String(zoneId)] || '#808080') : '#cccccc';

            return {
                fillColor: color,
                fillOpacity: 0.6,
                color: '#333',
                weight: 0.5,
            };
        },
        onEachFeature: (feature, layer) => {
            const bgId = String(feature.properties.BlockGroup);
            const zoneId = zones[bgId];
            const zoneDemographics = zoneId !== undefined ? zone_data[String(zoneId)] : null;
            const zoneIndex = zoneId !== undefined && zone_index_map ? zone_index_map[String(zoneId)] : null;

            const tooltipContent = createTooltip(bgId, zoneIndex, zoneDemographics);
            layer.bindTooltip(tooltipContent, { sticky: true });

            layer.on({
                mouseover: e => {
                    e.target.setStyle({ weight: 2, fillOpacity: 0.8 });
                },
                mouseout: e => {
                    geojsonLayer.resetStyle(e.target);
                },
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
        content += `<br>FRL: ${demographics.FRL_pct?.toFixed(1) || 0}%`;
    }

    return content;
}

function renderLegend() {
    const legendContainer = document.getElementById('zone-legend');
    if (!legendContainer || !currentSolution) {
        return;
    }

    const { colors, zone_index_map } = currentSolution;
    if (!colors || !zone_index_map) {
        legendContainer.classList.add('hidden');
        return;
    }

    // Build legend entries sorted by zone index
    const entries = Object.entries(zone_index_map)
        .map(([zoneId, index]) => ({
            zoneId,
            index,
            color: colors[zoneId] || '#808080'
        }))
        .sort((a, b) => a.index - b.index);

    let html = '<div class="legend-header"><h4>Zones</h4>';

    // Add school toggle button
    if (schoolMarkersLayer) {
        const buttonText = schoolsVisible ? 'Hide Schools' : 'Show Schools';
        const buttonClass = schoolsVisible ? 'active' : '';
        html += `<button id="toggle-schools-btn" class="toggle-schools-btn ${buttonClass}" title="Toggle school markers">${buttonText}</button>`;
    }

    html += '</div>';

    for (const entry of entries) {
        html += `<div class="legend-item">
            <div class="legend-color" style="background-color: ${entry.color}"></div>
            <span class="legend-label">Zone ${entry.index}</span>
        </div>`;
    }

    legendContainer.innerHTML = html;
    legendContainer.classList.remove('hidden');

    // Attach event listener to toggle button
    const toggleBtn = document.getElementById('toggle-schools-btn');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', toggleSchoolMarkers);
    }
}

function toggleSchoolMarkers() {
    if (!schoolMarkersLayer || !map) return;

    schoolsVisible = !schoolsVisible;

    if (schoolsVisible) {
        map.addLayer(schoolMarkersLayer);
    } else {
        map.removeLayer(schoolMarkersLayer);
    }

    // Update button text and style
    const toggleBtn = document.getElementById('toggle-schools-btn');
    if (toggleBtn) {
        toggleBtn.textContent = schoolsVisible ? 'Hide Schools' : 'Show Schools';
        toggleBtn.classList.toggle('active', schoolsVisible);
    }

    trackEvent('schools_toggled', { visible: schoolsVisible });
}

function refreshSingleChart() {
    // Re-render the currently selected metric chart with new solution data
    if (selectedMetricKey && currentSolution && currentSolution.zone_data) {
        showSingleChart(selectedMetricKey);
    }
}

// ============================================================================
// Solution History Management
// ============================================================================

let historyExpanded = false;

// Category percentile helpers — compute average percentile per category from solution data
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

function getCategoryPercentile(ranks, categoryMetrics) {
    if (!ranks) return null;
    let sum = 0, count = 0;
    for (const m of categoryMetrics) {
        const r = ranks[m.key];
        if (r && r.percentile !== undefined) {
            sum += r.percentile;
            count++;
        }
    }
    return count > 0 ? Math.round(sum / count) : null;
}

function autoSaveSolution(solutionData, label, agentMessage) {
    // Don't save duplicates (same solution path)
    const path = solutionData.path || '';
    if (path && savedSolutions.some(s => s.path === path)) {
        const existing = savedSolutions.find(s => s.path === path);
        if (existing) {
            currentViewedIndex = existing.index;
            renderSolutionHistory();
        }
        return;
    }

    // Enforce max cap
    if (savedSolutions.length >= MAX_SAVED_SOLUTIONS) {
        savedSolutions.shift();
        savedSolutions.forEach((s, i) => { s.index = i + 1; });
    }

    const index = savedSolutions.length + 1;

    // Pre-compute category percentiles for preview badges
    const ranks = solutionData.percentile_ranks || {};
    const categoryScores = {};
    for (const [cat, metrics] of Object.entries(HISTORY_CATEGORIES)) {
        categoryScores[cat] = getCategoryPercentile(ranks, metrics);
    }

    const entry = {
        index,
        path,
        solutionData: JSON.parse(JSON.stringify(solutionData)),
        label: label || `Solution #${index}`,
        agentMessage: agentMessage || '',
        pros: '',
        cons: '',
        timestamp: new Date().toISOString(),
        categoryScores,
    };

    savedSolutions.push(entry);
    currentViewedIndex = index;
    renderSolutionHistory();

    trackEvent('solution_saved', {
        solution_index: index,
        label: entry.label,
        solution_path: path,
    });
}

function viewSavedSolution(index) {
    const entry = savedSolutions.find(s => s.index === index);
    if (!entry) return;

    currentViewedIndex = index;
    currentSolution = entry.solutionData;

    // Re-render all views from cached data (no API call)
    loadGeojson().then(geojson => {
        renderMap(geojson);
        renderLegend();
        updateComparisonTable();
        refreshSingleChart();
        mapPlaceholder.classList.add('hidden');
    });

    renderSolutionHistory();
    updateProsConsPanel();

    trackEvent('solution_revisited', {
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

    // Invalidate map size after panel resize
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

    // Show and set initial state
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

        // Top row: number + label + note button
        const top = document.createElement('div');
        top.className = 'solution-card-top';

        const number = document.createElement('span');
        number.className = 'solution-card-number';
        number.textContent = entry.index;

        const label = document.createElement('span');
        label.className = 'solution-card-label';
        label.textContent = entry.label;
        label.title = entry.label;

        const noteBtn = document.createElement('button');
        noteBtn.className = 'solution-card-note-btn';
        noteBtn.innerHTML = '&#9998;';
        noteBtn.title = (entry.pros || entry.cons) ? 'Edit pros/cons' : 'Add pros/cons';
        noteBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (!historyExpanded) toggleHistoryExpanded();
            currentViewedIndex = entry.index;
            viewSavedSolution(entry.index);
            setTimeout(() => {
                const textarea = document.getElementById('solution-pros-textarea');
                if (textarea) textarea.focus();
            }, 100);
        });

        top.appendChild(number);
        top.appendChild(label);
        top.appendChild(noteBtn);
        card.appendChild(top);

        // Metric preview badges (visible only when expanded via CSS)
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

    // Auto-scroll to the active card
    const activeCard = cardsContainer.querySelector('.solution-card.active');
    if (activeCard) {
        activeCard.scrollIntoView({ behavior: 'smooth', inline: 'nearest', block: 'nearest' });
    }

    // Wire up toggle button (in case it was re-rendered)
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
        fresh.addEventListener('blur', () => saveProsCons(entry.index, field, fresh.value.trim()));
        fresh.addEventListener('keydown', (e) => { if (e.key === 'Escape') fresh.blur(); });
    }
}

function saveProsCons(index, field, text) {
    const entry = savedSolutions.find(s => s.index === index);
    if (!entry || entry[field] === text) return;
    entry[field] = text;
    renderSolutionHistory();
    trackEvent('solution_proscons_updated', { solution_index: index, field, length: text.length });
}

function buildSavedSolutionsSummary() {
    // Build a lightweight summary for the API payload
    return savedSolutions.map(s => {
        const metrics = s.solutionData.metrics || {};
        return {
            index: s.index,
            label: s.label,
            pros: s.pros,
            cons: s.cons,
            key_metrics: {
                frl: metrics.FRL,
                diversity: metrics.theil_index,
                distance: metrics.avg_closest_zone_school_distance,
                programs: metrics.avg_total_programs_per_zone,
            },
        };
    });
}

function updateComparisonTable() {
    const container = document.getElementById('comparison-table-container');
    const ranks = currentSolution && currentSolution.percentile_ranks;

    if (!currentSolution || !currentSolution.metrics || !ranks) {
        container.innerHTML = '<p class="no-solution-msg">Select a solution to see comparison</p>';
        return;
    }

    const metrics = currentSolution.metrics;

    // Define metric categories and their metrics
    const categories = {
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
            // { key: 'avg_greatschools_rating', name: 'GreatSchools' },
            { key: 'avg_math_score', name: 'Math Scores' },
            { key: 'avg_eng_score', name: 'English Scores' },
            // { key: 'avg_suspension_index', name: 'Suspension Index' },
        ],
    };

    let html = '<div class="category-list">';

    for (const [category, metricList] of Object.entries(categories)) {
        const catId = category.toLowerCase();

        // Compute average percentile for this category
        let percentileSum = 0;
        let percentileCount = 0;
        for (const { key } of metricList) {
            const rank = ranks[key];
            if (rank && rank.percentile !== undefined) {
                percentileSum += rank.percentile;
                percentileCount++;
            }
        }
        const avgPercentile = percentileCount > 0 ? Math.round(percentileSum / percentileCount) : null;
        const avgRanking = avgPercentile !== null ? getPercentileRanking(avgPercentile) : 'average';
        const avgBadgeHtml = avgPercentile !== null
            ? `<span class="percentile-indicator ${avgRanking}">${avgPercentile}%</span>`
            : '';

        html += `<div class="category-card collapsed" data-category="${catId}" data-avg-badge="${encodeURIComponent(avgBadgeHtml)}">`;
        html += `<div class="category-card-header">`;
        html += `<span class="category-card-title"><span class="chevron">&#9654;</span> ${category}</span>`;
        html += `<span class="category-avg-rank">${avgBadgeHtml}</span>`;
        html += `</div>`;

        html += `<table class="comparison-table category-metrics hidden">`;
        for (const { key, name } of metricList) {
            const value = metrics[key];
            const rank = ranks[key];
            if (value === undefined || !rank) continue;

            const displayPercentile = `${rank.percentile}%`;
            const clickable = METRIC_CHART_CONFIG[key] && METRIC_CHART_CONFIG[key].type !== 'none';
            const selectedClass = key === selectedMetricKey ? ' selected' : '';

            html += `<tr class="metric-row${selectedClass}" data-key="${key}"${clickable ? ' data-clickable="true"' : ''}>`;
            html += `<td class="metric-name">${name}</td>`;
            html += `<td class="metric-rank"><span class="percentile-indicator ${rank.ranking}">${displayPercentile}</span></td>`;
            html += `</tr>`;
        }
        html += `</table>`;
        html += `</div>`;
    }

    html += '</div>';
    container.innerHTML = html;

    // Attach click handlers for category card headers
    container.querySelectorAll('.category-card').forEach(card => {
        const header = card.querySelector('.category-card-header');
        const metricsTable = card.querySelector('.category-metrics');

        header.addEventListener('click', () => {
            const isCollapsed = card.classList.contains('collapsed');

            if (isCollapsed) {
                card.classList.remove('collapsed');
                card.classList.add('expanded');
                card.querySelector('.chevron').innerHTML = '&#9660;';
                card.querySelector('.category-avg-rank').innerHTML = '';
                metricsTable.classList.remove('hidden');
            } else {
                card.classList.remove('expanded');
                card.classList.add('collapsed');
                card.querySelector('.chevron').innerHTML = '&#9654;';
                card.querySelector('.category-avg-rank').innerHTML = decodeURIComponent(card.dataset.avgBadge);
                metricsTable.classList.add('hidden');
            }
        });
    });

    // Attach click handlers for metric rows
    container.querySelectorAll('.metric-row[data-clickable="true"]').forEach(row => {
        row.addEventListener('click', () => {
            showSingleChart(row.dataset.key);
        });
    });

    trackEvent('comparison_table_viewed', {
        num_metrics: Object.keys(metrics).length,
        solution_path: currentSolution.path,
    });
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

    // Format based on metric type
    if (key.includes('distance')) {
        return value.toFixed(2) + ' miles';
    }
    if (key.includes('rating')) {
        return value.toFixed(1);
    }
    if (key === 'boundary_cost') {
        return Math.round(value).toString();
    }
    if (key.includes('index') || key.includes('Index')) {
        return value.toFixed(2);
    }
    return value.toFixed(2);
}

function setupEventListeners() {
    console.log('[setupEventListeners] Setting up event listeners');
    console.log('[setupEventListeners] chatSend:', chatSend);
    console.log('[setupEventListeners] chatInput:', chatInput);

    if (chatSend) {
        chatSend.addEventListener('click', () => {
            console.log('[chatSend] Button clicked');
            sendMessage();
        });
    } else {
        console.error('[setupEventListeners] chatSend element not found!');
    }

    if (chatInput) {
        chatInput.addEventListener('keypress', e => {
            if (e.key === 'Enter' && !isProcessing) {
                console.log('[chatInput] Enter key pressed');
                sendMessage();
            }
        });
    } else {
        console.error('[setupEventListeners] chatInput element not found!');
    }

    const chartsClose = document.getElementById('charts-close');
    if (chartsClose) {
        chartsClose.addEventListener('click', () => {
            document.getElementById('charts-panel').classList.add('hidden');
            // Deselect metric row
            document.querySelectorAll('.metric-row.selected').forEach(r => r.classList.remove('selected'));
            selectedMetricKey = null;
            if (singleChart) {
                singleChart.destroy();
                singleChart = null;
            }
        });
    }

    setupResizeHandle();
}

function setupResizeHandle() {
    let isResizing = false;
    let startX = 0;
    let startY = 0;
    let startChatWidth = 0;

    if (resizeHandleRight) {
        resizeHandleRight.addEventListener('mousedown', e => {
            isResizing = true;
            startX = e.clientX;
            startY = e.clientY;
            startChatWidth = document.getElementById('chat-panel').offsetWidth;
            resizeHandleRight.classList.add('resizing');

            const isMobileView = window.innerWidth <= 900;
            document.body.style.cursor = isMobileView ? 'row-resize' : 'col-resize';
            document.body.style.userSelect = 'none';
            e.preventDefault();
        });
    }

    document.addEventListener('mousemove', e => {
        if (!isResizing) return;

        const isMobileView = window.innerWidth <= 900;

        if (isMobileView) {
            const deltaY = startY - e.clientY;
            const newHeight = Math.max(200, Math.min(600, startChatWidth + deltaY));
            mainContainer.style.gridTemplateRows = `1fr 4px ${newHeight}px`;
        } else {
            const deltaX = e.clientX - startX;
            const newChatWidth = Math.max(250, Math.min(600, startChatWidth - deltaX));
            mainContainer.style.gridTemplateColumns = `1fr 4px ${newChatWidth}px`;
        }

        if (map) {
            map.invalidateSize();
        }
    });

    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            resizeHandleRight.classList.remove('resizing');
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        }
    });
}

async function sendInitialMessage() {
    await sendMessageToAgent('Show me the available zoning options grouped by their trade-offs.');
}

async function sendMessage() {
    const message = chatInput.value.trim();
    console.log('[sendMessage] Called, message:', message, 'isProcessing:', isProcessing);

    if (!message) {
        console.log('[sendMessage] Empty message, ignoring');
        return;
    }

    if (isProcessing) {
        console.log('[sendMessage] Already processing, ignoring');
        return;
    }

    addMessage('user', message);
    chatInput.value = '';

    trackEvent('chat_message_sent', {
        message_text: message,
        message_length: message.length,
        session_id: sessionId,
    });

    await sendMessageToAgent(message);
}

async function sendMessageToAgent(message) {
    console.log('[sendMessageToAgent] Called with message:', message);
    setProcessing(true);
    showMapLoading(true);

    let thinkingMsg;
    try {
        thinkingMsg = addMessage('loading', 'Thinking... (this may take a moment)');
    } catch (e) {
        console.error('[sendMessageToAgent] Failed to add thinking message:', e);
    }

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 120000);

        const response = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                session_id: sessionId,
                current_solution_index: currentViewedIndex,
                saved_solutions: buildSavedSolutionsSummary(),
            }),
            signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Chat response error:', response.status, errorText);
            throw new Error(`Chat request failed: ${response.status}`);
        }

        const data = await response.json();
        console.log('Chat response:', data);

        sessionId = data.session_id;

        if (thinkingMsg) thinkingMsg.remove();

        trackEvent('agent_response_received', {
            response_type: data.response_type,
            response_text: data.text,
            has_clusters: !!(data.clusters && data.clusters.length > 0),
            has_solution: !!data.solution_path,
            session_id: sessionId,
        });

        if (data.response_type === 'clusters' && data.clusters && data.clusters.length > 0) {
            if (data.text) {
                addMessage('assistant', data.text);
            }
            renderClusterSelector(data.clusters);
        } else if (data.response_type === 'solution_update' && data.solution_path) {
            if (data.text) {
                addMessage('assistant', data.text);
            }
            await loadSolution(data.solution_path);
            // Auto-save the solution after loading
            if (currentSolution) {
                const label = data.description || data.text?.substring(0, 50) || 'Solution';
                autoSaveSolution(currentSolution, label, data.text);
            }
        } else {
            addMessage('assistant', data.text || 'Agent returned empty response. Please try again.');
        }
    } catch (error) {
        console.error('Chat error:', error);
        if (thinkingMsg) thinkingMsg.remove();
        if (error.name === 'AbortError') {
            addMessage('assistant', 'Request timed out. The agent is taking too long to respond. Please try again.');
        } else {
            addMessage('assistant', `Error: ${error.message}. Please try again.`);
        }
    } finally {
        setProcessing(false);
        showMapLoading(false);
    }
}

function renderClusterSelector(clusters) {
    const container = document.createElement('div');
    container.className = 'cluster-selector';

    const title = document.createElement('div');
    title.className = 'cluster-selector-title';
    title.textContent = 'Select a zoning approach:';
    container.appendChild(title);

    clusters.forEach(cluster => {
        const option = document.createElement('div');
        option.className = 'cluster-option';
        option.onclick = () => selectCluster(cluster.id, cluster.label);

        const label = document.createElement('div');
        label.className = 'cluster-option-label';
        label.textContent = `${cluster.id}. ${cluster.label}`;

        const meta = document.createElement('div');
        meta.className = 'cluster-option-meta';
        meta.textContent = `${cluster.count} solutions`;

        option.appendChild(label);
        option.appendChild(meta);
        container.appendChild(option);
    });

    const wrapper = document.createElement('div');
    wrapper.className = 'message assistant';
    wrapper.appendChild(container);

    chatMessages.appendChild(wrapper);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function selectCluster(clusterId, clusterLabel) {
    const allClusterSelectors = document.querySelectorAll('.cluster-selector');
    allClusterSelectors.forEach(selector => {
        selector.classList.add('disabled');
        const options = selector.querySelectorAll('.cluster-option');
        options.forEach(opt => {
            opt.onclick = null;
            opt.style.cursor = 'default';
        });
    });

    trackEvent('cluster_selected', {
        cluster_id: clusterId,
        cluster_label: clusterLabel,
    });

    addMessage('user', `Select cluster ${clusterId}: ${clusterLabel}`);
    await sendMessageToAgent(`Select cluster ${clusterId}`);
}

function addMessage(type, content) {
    const div = document.createElement('div');
    div.className = `message ${type}`;

    if (type === 'assistant' && typeof marked !== 'undefined') {
        marked.setOptions({
            breaks: true,
            gfm: true,
        });
        div.innerHTML = marked.parse(content);
    } else {
        div.textContent = content;
    }

    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return div;
}

function setProcessing(processing) {
    console.log('[setProcessing]', processing);
    isProcessing = processing;
    if (chatInputArea) {
        chatInputArea.classList.toggle('processing', processing);
    }
}

function showMapLoading(show) {
    mapLoadingOverlay.classList.toggle('hidden', !show);
}

function trackEvent(eventName, properties) {
    if (typeof posthog !== 'undefined' && posthog.capture) {
        posthog.capture(eventName, properties);
    }
}

function showLoading(show) {
    loadingOverlay.classList.toggle('hidden', !show);
}
