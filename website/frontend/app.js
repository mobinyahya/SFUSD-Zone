// SFUSD Zoning Dashboard - Main Application

const API_BASE = '';

// State
let map = null;
let geojsonLayer = null;
let geojsonData = null;
let currentSolution = null;
let currentClusterLabel = '';
let demographicsChart = null;
let studentsChart = null;
let clusters = [];

// DOM Elements
const clusterDropdown = document.getElementById('cluster-dropdown');
const mapPlaceholder = document.getElementById('map-placeholder');
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const chatSend = document.getElementById('chat-send');
const loadingOverlay = document.getElementById('loading-overlay');

// Initialize
document.addEventListener('DOMContentLoaded', init);

async function init() {
    initMap();
    initCharts();
    await loadClusters();
    setupEventListeners();
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

async function loadClusters() {
    try {
        const response = await fetch(`${API_BASE}/api/clusters`);
        if (!response.ok) throw new Error('Failed to load clusters');

        const data = await response.json();
        clusters = data.clusters;

        clusterDropdown.innerHTML = '<option value="">-- Select a zoning approach --</option>';
        clusters.forEach(cluster => {
            const option = document.createElement('option');
            option.value = cluster.path;
            option.textContent = `${cluster.label} (${cluster.count} solutions)`;
            option.dataset.clusterId = cluster.id;
            option.dataset.label = cluster.label;
            clusterDropdown.appendChild(option);
        });

        clusterDropdown.disabled = false;
    } catch (error) {
        console.error('Error loading clusters:', error);
        clusterDropdown.innerHTML = '<option value="">Error loading clusters</option>';
        addMessage('system', 'Failed to load zoning clusters. Please refresh the page.');
    }
}

async function loadGeojson() {
    if (geojsonData) return geojsonData;

    const response = await fetch(`${API_BASE}/api/geojson`);
    if (!response.ok) throw new Error('Failed to load GeoJSON');

    geojsonData = await response.json();
    return geojsonData;
}

async function loadSolution(path, clusterLabel) {
    showLoading(true);

    try {
        const [geojson, solutionResponse] = await Promise.all([
            loadGeojson(),
            fetch(`${API_BASE}/api/solution/${encodeURIComponent(path)}`)
        ]);

        if (!solutionResponse.ok) throw new Error('Failed to load solution');

        currentSolution = await solutionResponse.json();
        currentClusterLabel = clusterLabel;

        renderMap(geojson);
        updateCharts();
        enableChat();

        mapPlaceholder.classList.add('hidden');
        addMessage('system', `Loaded: ${clusterLabel}`);
    } catch (error) {
        console.error('Error loading solution:', error);
        addMessage('system', 'Failed to load solution. Please try again.');
    } finally {
        showLoading(false);
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
    if (!currentSolution) return;

    const { demographics } = currentSolution;
    const zoneIds = Object.keys(demographics).sort((a, b) => Number(a) - Number(b));

    // FRL chart
    const frlData = zoneIds.map(id => demographics[id].FRL_pct || 0);
    demographicsChart.data.labels = zoneIds.map(id => `Zone ${id}`);
    demographicsChart.data.datasets[0].data = frlData;
    demographicsChart.update();

    // Students chart
    const studentData = zoneIds.map(id => Math.round(demographics[id].ge_students || 0));
    studentsChart.data.labels = zoneIds.map(id => `Zone ${id}`);
    studentsChart.data.datasets[0].data = studentData;
    studentsChart.update();
}

function enableChat() {
    chatInput.disabled = false;
    chatSend.disabled = false;
}

function setupEventListeners() {
    clusterDropdown.addEventListener('change', async e => {
        const path = e.target.value;
        if (!path) return;

        const selected = e.target.options[e.target.selectedIndex];
        const label = selected.dataset.label || 'Selected solution';
        await loadSolution(path, label);
    });

    chatSend.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', e => {
        if (e.key === 'Enter') sendMessage();
    });
}

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    addMessage('user', message);
    chatInput.value = '';

    try {
        const response = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                cluster_label: currentClusterLabel,
            }),
        });

        if (!response.ok) throw new Error('Chat request failed');

        const data = await response.json();
        addMessage('assistant', data.response);
    } catch (error) {
        console.error('Chat error:', error);
        addMessage('assistant', 'Sorry, there was an error processing your message.');
    }
}

function addMessage(type, content) {
    const div = document.createElement('div');
    div.className = `message ${type}`;
    div.textContent = content;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showLoading(show) {
    loadingOverlay.classList.toggle('hidden', !show);
}
