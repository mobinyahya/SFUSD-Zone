// SFUSD Zoning Dashboard - Main Application

const API_BASE = '';

// State
let map = null;
let geojsonLayer = null;
let geojsonData = null;
let currentSolution = null;
let solutionSpaceStats = null;
let sessionId = null;
let isProcessing = false;

// Chart instances
let charts = {};

// DOM Elements
const mapPlaceholder = document.getElementById('map-placeholder');
const mapLoadingOverlay = document.getElementById('map-loading-overlay');
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const chatSend = document.getElementById('chat-send');
const chatInputArea = document.getElementById('chat-input-area');
const loadingOverlay = document.getElementById('loading-overlay');
const resizeHandleLeft = document.getElementById('resize-handle-left');
const resizeHandleRight = document.getElementById('resize-handle-right');
const mainContainer = document.querySelector('main');
const comparisonPanel = document.getElementById('comparison-panel');

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
    // Set up event listeners FIRST - these are critical and should work even if other things fail
    setupEventListeners();
    try {
        initMap();
        setupTabSwitching();
    } catch (error) {
        console.error('Failed to initialize map:', error);
    }

    try {
        initCharts();
    } catch (error) {
        console.error('Failed to initialize charts:', error);
    }

    // Load solution space stats for comparison table
    try {
        await loadSolutionSpaceStats();
    } catch (error) {
        console.error('Failed to load solution space stats:', error);
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
}

function initCharts() {
    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false }
        },
        scales: {
            y: { beginAtZero: true }
        }
    };

    // Diversity Tab Charts
    charts.frl = new Chart(document.getElementById('chart-frl').getContext('2d'), {
        type: 'bar',
        data: { labels: [], datasets: [{ label: 'FRL %', data: [], backgroundColor: CHART_COLORS.primary }] },
        options: { ...defaultOptions, scales: { y: { beginAtZero: true, max: 100, title: { display: true, text: '%' } } } }
    });

    charts.ethnicity = new Chart(document.getElementById('chart-ethnicity').getContext('2d'), {
        type: 'bar',
        data: {
            labels: [],
            datasets: [
                { label: 'Black/African American', data: [], backgroundColor: CHART_COLORS.ethnicities['Black/African American'] },
                { label: 'Hispanic/Latinx', data: [], backgroundColor: CHART_COLORS.ethnicities['Hispanic/Latinx'] },
                { label: 'White', data: [], backgroundColor: CHART_COLORS.ethnicities['White'] },
                { label: 'Asian', data: [], backgroundColor: CHART_COLORS.ethnicities['Asian'] },
                { label: 'Other', data: [], backgroundColor: CHART_COLORS.ethnicities['Other'] },
            ]
        },
        options: {
            ...defaultOptions,
            plugins: { legend: { display: true, position: 'bottom', labels: { boxWidth: 12, font: { size: 10 } } } },
            scales: { x: { stacked: true }, y: { stacked: true, max: 100, title: { display: true, text: '%' } } }
        }
    });

    // Distance Tab Charts
    charts.distance = new Chart(document.getElementById('chart-distance').getContext('2d'), {
        type: 'bar',
        data: { labels: [], datasets: [{ label: 'Avg Distance', data: [], backgroundColor: CHART_COLORS.secondary }] },
        options: { ...defaultOptions, scales: { y: { beginAtZero: true, title: { display: true, text: 'km' } } } }
    });

    charts.attendance = new Chart(document.getElementById('chart-attendance').getContext('2d'), {
        type: 'bar',
        data: { labels: [], datasets: [{ label: 'Schools', data: [], backgroundColor: CHART_COLORS.primary }] },
        options: { ...defaultOptions, scales: { y: { beginAtZero: true, title: { display: true, text: 'Count' } } } }
    });

    charts.students = new Chart(document.getElementById('chart-students').getContext('2d'), {
        type: 'bar',
        data: { labels: [], datasets: [{ label: 'Students', data: [], backgroundColor: CHART_COLORS.quinary }] },
        options: { ...defaultOptions, scales: { y: { beginAtZero: true, title: { display: true, text: 'Count' } } } }
    });

    // Programs Tab Charts
    charts.totalPrograms = new Chart(document.getElementById('chart-total-programs').getContext('2d'), {
        type: 'bar',
        data: { labels: [], datasets: [{ label: 'Total Programs', data: [], backgroundColor: CHART_COLORS.primary }] },
        options: { ...defaultOptions, scales: { y: { beginAtZero: true, title: { display: true, text: 'Count' } } } }
    });

    charts.languageImmersion = new Chart(document.getElementById('chart-language-immersion').getContext('2d'), {
        type: 'bar',
        data: { labels: [], datasets: [{ label: 'Language Immersion', data: [], backgroundColor: CHART_COLORS.tertiary }] },
        options: { ...defaultOptions, scales: { y: { beginAtZero: true, title: { display: true, text: 'Count' } } } }
    });

    charts.specialEd = new Chart(document.getElementById('chart-special-ed').getContext('2d'), {
        type: 'bar',
        data: { labels: [], datasets: [{ label: 'Special Ed', data: [], backgroundColor: CHART_COLORS.quaternary }] },
        options: { ...defaultOptions, scales: { y: { beginAtZero: true, title: { display: true, text: 'Count' } } } }
    });

    // Quality Tab Charts
    charts.greatschools = new Chart(document.getElementById('chart-greatschools').getContext('2d'), {
        type: 'bar',
        data: { labels: [], datasets: [{ label: 'Rating', data: [], backgroundColor: CHART_COLORS.secondary }] },
        options: { ...defaultOptions, scales: { y: { beginAtZero: true, max: 10, title: { display: true, text: 'Rating' } } } }
    });

    charts.scores = new Chart(document.getElementById('chart-scores').getContext('2d'), {
        type: 'bar',
        data: {
            labels: [],
            datasets: [
                { label: 'Math', data: [], backgroundColor: CHART_COLORS.primary },
                { label: 'English', data: [], backgroundColor: CHART_COLORS.tertiary },
            ]
        },
        options: {
            ...defaultOptions,
            plugins: { legend: { display: true, position: 'bottom', labels: { boxWidth: 12, font: { size: 10 } } } },
            scales: { y: { beginAtZero: true, title: { display: true, text: 'Score' } } }
        }
    });

    charts.suspension = new Chart(document.getElementById('chart-suspension').getContext('2d'), {
        type: 'bar',
        data: { labels: [], datasets: [{ label: 'Index', data: [], backgroundColor: CHART_COLORS.quinary }] },
        options: { ...defaultOptions, scales: { y: { beginAtZero: true, max: 5, title: { display: true, text: 'Index' } } } }
    });
}

function setupTabSwitching() {
    const tabs = document.querySelectorAll('.chart-tab');
    const panes = document.querySelectorAll('.tab-pane');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetTab = tab.dataset.tab;

            // Update active tab
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            // Update active pane
            panes.forEach(pane => {
                pane.classList.toggle('active', pane.dataset.tab === targetTab);
            });

            // Trigger chart resize for visible charts
            setTimeout(() => {
                Object.values(charts).forEach(chart => chart.resize());
            }, 100);
        });
    });
}

async function loadSolutionSpaceStats() {
    try {
        const response = await fetch(`${API_BASE}/api/solution-space-stats`);
        if (!response.ok) {
            console.warn('Failed to load solution space stats');
            return;
        }
        const data = await response.json();
        solutionSpaceStats = data.stats;
        console.log('Solution space stats loaded:', Object.keys(solutionSpaceStats).length, 'metrics');
    } catch (error) {
        console.error('Error loading solution space stats:', error);
    }
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

        renderMap(geojson);
        updateAllCharts();
        updateComparisonTable();

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

    const { zones, demographics, colors } = currentSolution;

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
            const zoneDemographics = zoneId !== undefined ? demographics[String(zoneId)] : null;

            const tooltipContent = createTooltip(bgId, zoneId, zoneDemographics);
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

function createTooltip(bgId, zoneId, demographics) {
    let content = `<strong>BlockGroup: ${bgId}</strong>`;

    if (zoneId !== undefined) {
        content += `<br><span class="zone-info">Zone: ${zoneId}</span>`;
    }

    if (demographics) {
        content += `<br>Students: ${Math.round(demographics.ge_students)}`;
        content += `<br>FRL: ${demographics.FRL_pct?.toFixed(1) || 0}%`;
    }

    return content;
}

function updateAllCharts() {
    if (!currentSolution || !currentSolution.zone_data) {
        console.warn('[updateAllCharts] No solution data available');
        return;
    }

    const zoneData = currentSolution.zone_data;
    const zoneIds = Object.keys(zoneData).sort((a, b) => Number(a) - Number(b));
    // Use sequential zone labels (Zone 0, Zone 1, etc.) instead of actual zone IDs
    const labels = zoneIds.map((id, index) => `Zone ${index}`);

    // Diversity Charts
    updateChart(charts.frl, labels, zoneIds.map(id => zoneData[id].FRL_pct || (zoneData[id].frl_pct * 100) || 0));

    // Ethnicity stacked bar
    const ethnicityKeys = ['Ethnicity_Black_or_African_American', 'Ethnicity_Hispanic/Latinx', 'Ethnicity_White', 'Ethnicity_Asian'];
    const ethnicityData = ethnicityKeys.map(key =>
        zoneIds.map(id => {
            const pcts = zoneData[id].ethnicity_pcts || {};
            return (pcts[key] || 0) * 100;
        })
    );
    // Calculate "Other" as remainder
    const otherData = zoneIds.map((id, idx) => {
        const sum = ethnicityData.reduce((acc, arr) => acc + arr[idx], 0);
        return Math.max(0, 100 - sum);
    });

    charts.ethnicity.data.labels = labels;
    charts.ethnicity.data.datasets[0].data = ethnicityData[0];
    charts.ethnicity.data.datasets[1].data = ethnicityData[1];
    charts.ethnicity.data.datasets[2].data = ethnicityData[2];
    charts.ethnicity.data.datasets[3].data = ethnicityData[3];
    charts.ethnicity.data.datasets[4].data = otherData;
    charts.ethnicity.update();

    // Distance Charts
    updateChart(charts.distance, labels, zoneIds.map(id => zoneData[id].avg_closest_school_distance || 0));
    updateChart(charts.attendance, labels, zoneIds.map(id => zoneData[id].schools_in_attendance_area || 0));
    updateChart(charts.students, labels, zoneIds.map(id => Math.round(zoneData[id].ge_students || 0)));

    // Programs Charts
    updateChart(charts.totalPrograms, labels, zoneIds.map(id => zoneData[id].total_programs || 0));
    updateChart(charts.languageImmersion, labels, zoneIds.map(id => zoneData[id].language_immersion_count || 0));
    updateChart(charts.specialEd, labels, zoneIds.map(id => zoneData[id].special_ed_count || 0));

    // Quality Charts
    updateChart(charts.greatschools, labels, zoneIds.map(id => zoneData[id].avg_greatschools_rating || 0));

    // Test scores
    const mathScores = zoneIds.map(id => zoneData[id].avg_math_score || 0);
    const engScores = zoneIds.map(id => zoneData[id].avg_eng_score || 0);
    charts.scores.data.labels = labels;
    charts.scores.data.datasets[0].data = mathScores;
    charts.scores.data.datasets[1].data = engScores;
    charts.scores.update();

    updateChart(charts.suspension, labels, zoneIds.map(id => zoneData[id].avg_suspension_index || 0));
}

function updateChart(chart, labels, data) {
    chart.data.labels = labels;
    chart.data.datasets[0].data = data;
    chart.update();
}

function updateComparisonTable() {
    const container = document.getElementById('comparison-table-container');

    if (!currentSolution || !currentSolution.metrics || !solutionSpaceStats) {
        container.innerHTML = '<p class="no-solution-msg">Select a solution to see comparison</p>';
        return;
    }

    const metrics = currentSolution.metrics;

    // Define metric categories and their metrics
    const categories = {
        'Diversity': [
            { key: 'theil_index', name: 'Ethnic Segregation' },
            { key: 'FRL', name: 'FRL Deviation' },
            { key: 'seat_disparity', name: 'Seat Disparity' },
        ],
        'Distance': [
            { key: 'avg_closest_zone_school_distance', name: 'Avg Distance' },
            { key: 'avg_schools_in_attendance_area', name: 'Schools in Area' },
            { key: 'boundary_cost', name: 'Boundary Cost' },
        ],
        'Programs': [
            { key: 'avg_total_programs_per_zone', name: 'Total Programs' },
            { key: 'avg_language_immersion_per_zone', name: 'Language Immersion' },
            { key: 'avg_special_ed_per_zone', name: 'Special Ed' },
        ],
        'Quality': [
            { key: 'avg_greatschools_rating', name: 'GreatSchools' },
            { key: 'avg_math_score', name: 'Math Scores' },
            { key: 'avg_eng_score', name: 'English Scores' },
            { key: 'avg_suspension_index', name: 'Suspension Index' },
        ],
    };

    let html = '<table class="comparison-table">';
    html += '<thead><tr><th>Metric</th><th>Value</th><th>Rank</th></tr></thead>';
    html += '<tbody>';

    for (const [category, metricList] of Object.entries(categories)) {
        html += `<tr class="category-header"><td colspan="3">${category}</td></tr>`;

        for (const { key, name } of metricList) {
            const value = metrics[key];
            const stats = solutionSpaceStats[key];

            if (value === undefined || !stats) continue;

            const percentile = calculatePercentile(value, stats);
            const ranking = getRankingClass(percentile, stats.direction);
            const displayValue = formatValue(value, key);
            const displayPercentile = `${Math.round(percentile)}%`;

            html += `<tr>`;
            html += `<td class="metric-name">${name}</td>`;
            html += `<td class="metric-value">${displayValue}</td>`;
            html += `<td class="metric-rank"><span class="percentile-indicator ${ranking}">${displayPercentile}</span></td>`;
            html += `</tr>`;
        }
    }

    html += '</tbody></table>';
    container.innerHTML = html;
}

function calculatePercentile(value, stats) {
    // Calculate percentile based on where value falls in distribution
    const { min, max, p10, p25, p50, p75, p90 } = stats;

    if (value <= min) return 0;
    if (value >= max) return 100;

    // Linear interpolation between percentile points
    if (value <= p10) return 10 * (value - min) / (p10 - min);
    if (value <= p25) return 10 + 15 * (value - p10) / (p25 - p10);
    if (value <= p50) return 25 + 25 * (value - p25) / (p50 - p25);
    if (value <= p75) return 50 + 25 * (value - p50) / (p75 - p50);
    if (value <= p90) return 75 + 15 * (value - p75) / (p90 - p75);
    return 90 + 10 * (value - p90) / (max - p90);
}

function getRankingClass(percentile, direction) {
    // For "minimize" metrics, lower percentile is better
    // For "maximize" metrics, higher percentile is better
    const effectivePercentile = direction === 'minimize' ? (100 - percentile) : percentile;

    if (effectivePercentile >= 80) return 'good';
    if (effectivePercentile >= 20) return 'ok';
    return 'bad';
}

function formatValue(value, key) {
    if (value === undefined || value === null) return '-';

    // Format based on metric type
    if (key.includes('distance')) {
        return value.toFixed(2) + ' km';
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

    setupResizeHandle();
}

function setupResizeHandle() {
    let isResizing = false;
    let activeHandle = null;
    let startX = 0;
    let startY = 0;
    let startComparisonWidth = 0;
    let startChatWidth = 0;

    const startResize = (e, handle) => {
        isResizing = true;
        activeHandle = handle;
        startX = e.clientX;
        startY = e.clientY;
        startComparisonWidth = comparisonPanel.offsetWidth;
        startChatWidth = document.getElementById('chat-panel').offsetWidth;
        handle.classList.add('resizing');

        const isMobileView = window.innerWidth <= 900;
        document.body.style.cursor = isMobileView ? 'row-resize' : 'col-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();
    };

    if (resizeHandleLeft) {
        resizeHandleLeft.addEventListener('mousedown', e => startResize(e, resizeHandleLeft));
    }

    if (resizeHandleRight) {
        resizeHandleRight.addEventListener('mousedown', e => startResize(e, resizeHandleRight));
    }

    document.addEventListener('mousemove', e => {
        if (!isResizing) return;

        const isMobileView = window.innerWidth <= 900;

        if (isMobileView) {
            // Mobile: vertical resize for chat panel
            const deltaY = startY - e.clientY;
            const newHeight = Math.max(200, Math.min(600, startChatWidth + deltaY));
            mainContainer.style.gridTemplateRows = `1fr 4px ${newHeight}px`;
        } else {
            // Desktop: horizontal resize
            const deltaX = e.clientX - startX;

            if (activeHandle === resizeHandleLeft) {
                // Resizing comparison panel from left side
                const newComparisonWidth = Math.max(150, Math.min(500, startComparisonWidth - deltaX));
                const chatWidth = document.getElementById('chat-panel').offsetWidth;
                mainContainer.style.gridTemplateColumns = `1fr 4px ${newComparisonWidth}px 4px ${chatWidth}px`;
            } else if (activeHandle === resizeHandleRight) {
                // Resizing chat panel from left side
                const newChatWidth = Math.max(250, Math.min(600, startChatWidth - deltaX));
                mainContainer.style.gridTemplateColumns = `1fr 4px ${startComparisonWidth}px 4px ${newChatWidth}px`;
            }
        }

        if (map) {
            map.invalidateSize();
        }
    });

    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            if (activeHandle) {
                activeHandle.classList.remove('resizing');
            }
            activeHandle = null;
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

function showLoading(show) {
    loadingOverlay.classList.toggle('hidden', !show);
}
