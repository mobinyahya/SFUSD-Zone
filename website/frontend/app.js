// SFUSD Zoning Dashboard - Main Application

const API_BASE = '';

// State
let map = null;
let geojsonLayer = null;
let geojsonData = null;
let currentSolution = null;
let demographicsChart = null;
let studentsChart = null;
let sessionId = null;
let isProcessing = false;

// DOM Elements
const mapPlaceholder = document.getElementById('map-placeholder');
const mapLoadingOverlay = document.getElementById('map-loading-overlay');
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const chatSend = document.getElementById('chat-send');
const chatInputArea = document.getElementById('chat-input-area');
const loadingOverlay = document.getElementById('loading-overlay');
const resizeHandle = document.getElementById('resize-handle');
const mainContainer = document.querySelector('main');

// Initialize
document.addEventListener('DOMContentLoaded', init);

async function init() {
    initMap();
    initCharts();
    setupEventListeners();
    // Send initial message to trigger clustering
    await sendInitialMessage();
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
    const ctx1 = document.getElementById('demographics-chart').getContext('2d');
    demographicsChart = new Chart(ctx1, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'FRL %',
                data: [],
                backgroundColor: '#3498db',
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: { display: true, text: '%' }
                }
            }
        }
    });

    const ctx2 = document.getElementById('students-chart').getContext('2d');
    studentsChart = new Chart(ctx2, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Students',
                data: [],
                backgroundColor: '#2ecc71',
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: { display: true, text: 'Count' }
                }
            }
        }
    });
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
        updateCharts();

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

    // Ensure Leaflet recalculates container size before fitting bounds
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

function updateCharts() {
    console.log('[updateCharts] Called');
    if (!currentSolution) {
        console.warn('[updateCharts] No currentSolution, returning');
        return;
    }

    const { demographics } = currentSolution;
    console.log('[updateCharts] demographics:', demographics ? `${Object.keys(demographics).length} zones` : 'undefined');

    if (!demographics) {
        console.error('[updateCharts] demographics is undefined in currentSolution');
        return;
    }

    const zoneIds = Object.keys(demographics).sort((a, b) => Number(a) - Number(b));
    console.log('[updateCharts] zoneIds:', zoneIds);

    // FRL chart
    const frlData = zoneIds.map(id => {
        const value = demographics[id].FRL_pct || 0;
        console.log(`[updateCharts] Zone ${id} FRL_pct:`, value);
        return value;
    });
    demographicsChart.data.labels = zoneIds.map(id => `Zone ${id}`);
    demographicsChart.data.datasets[0].data = frlData;
    demographicsChart.update();
    console.log('[updateCharts] FRL chart updated with data:', frlData);

    // Students chart
    const studentData = zoneIds.map(id => Math.round(demographics[id].ge_students || 0));
    studentsChart.data.labels = zoneIds.map(id => `Zone ${id}`);
    studentsChart.data.datasets[0].data = studentData;
    studentsChart.update();
    console.log('[updateCharts] Students chart updated with data:', studentData);
}

function setupEventListeners() {
    chatSend.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', e => {
        if (e.key === 'Enter' && !isProcessing) sendMessage();
    });
    setupResizeHandle();
}

function setupResizeHandle() {
    let isResizing = false;
    let startX = 0;
    let startY = 0;
    let startWidth = 0;
    let startHeight = 0;

    resizeHandle.addEventListener('mousedown', e => {
        isResizing = true;
        startX = e.clientX;
        startY = e.clientY;
        const chatPanel = document.getElementById('chat-panel');
        startWidth = chatPanel.offsetWidth;
        startHeight = chatPanel.offsetHeight;
        resizeHandle.classList.add('resizing');

        // Detect if we're in mobile/vertical mode
        const isMobileView = window.innerWidth <= 900;
        document.body.style.cursor = isMobileView ? 'row-resize' : 'col-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();
    });

    document.addEventListener('mousemove', e => {
        if (!isResizing) return;

        // Detect if we're in mobile/vertical mode
        const isMobileView = window.innerWidth <= 900;

        if (isMobileView) {
            // Vertical resizing for mobile
            const deltaY = startY - e.clientY;
            const newHeight = Math.max(200, Math.min(600, startHeight + deltaY));
            mainContainer.style.gridTemplateRows = `1fr 4px ${newHeight}px`;
        } else {
            // Horizontal resizing for desktop
            const deltaX = startX - e.clientX;
            const newWidth = Math.max(250, Math.min(800, startWidth + deltaX));
            mainContainer.style.gridTemplateColumns = `1fr 4px ${newWidth}px`;
        }

        // Invalidate map size to ensure proper rendering after resize
        if (map) {
            map.invalidateSize();
        }
    });

    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            resizeHandle.classList.remove('resizing');
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        }
    });
}

async function sendInitialMessage() {
    // Automatically trigger clustering on load
    await sendMessageToAgent('Show me the available zoning options grouped by their trade-offs.');
}

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message || isProcessing) return;

    addMessage('user', message);
    chatInput.value = '';

    await sendMessageToAgent(message);
}

async function sendMessageToAgent(message) {
    setProcessing(true);
    showMapLoading(true);

    // Add thinking message
    const thinkingMsg = addMessage('loading', 'Thinking... (this may take a moment)');

    try {
        // Use AbortController for timeout (2 minutes for slow LLM calls)
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

        // Update session ID
        sessionId = data.session_id;

        // Remove thinking message
        thinkingMsg.remove();

        // Handle response based on type
        if (data.response_type === 'clusters' && data.clusters && data.clusters.length > 0) {
            // Show text response
            if (data.text) {
                addMessage('assistant', data.text);
            }
            // Render cluster selector in chat
            renderClusterSelector(data.clusters);
        } else if (data.response_type === 'solution_update' && data.solution_path) {
            // Show text response
            if (data.text) {
                addMessage('assistant', data.text);
            }
            // Load the new solution
            await loadSolution(data.solution_path);
        } else {
            // Just text response
            addMessage('assistant', data.text || 'Agent returned empty response. Please try again.');
        }
    } catch (error) {
        console.error('Chat error:', error);
        thinkingMsg.remove();
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
    // Disable all cluster selectors to prevent reselection
    const allClusterSelectors = document.querySelectorAll('.cluster-selector');
    allClusterSelectors.forEach(selector => {
        selector.classList.add('disabled');
        // Remove onclick handlers from all options
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

    // Render markdown for assistant messages
    if (type === 'assistant' && typeof marked !== 'undefined') {
        // Configure marked for safe rendering
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
    isProcessing = processing;
    chatInputArea.classList.toggle('processing', processing);
}

function showMapLoading(show) {
    mapLoadingOverlay.classList.toggle('hidden', !show);
}

function showLoading(show) {
    loadingOverlay.classList.toggle('hidden', !show);
}
