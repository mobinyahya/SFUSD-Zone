// SFUSD Zoning Dashboard - User Page (Chat Interface)
// Depends on shared.js being loaded first.

let posthogApiKey = null;
let sessionId = null;
let isProcessing = false;

const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const chatSend = document.getElementById('chat-send');
const chatInputArea = document.getElementById('chat-input-area');
const loadingOverlay = document.getElementById('loading-overlay');

// ============================================================================
// Page Hooks (configure shared.js behavior for user page)
// ============================================================================

pageHooks.rightPanelSelector = '#chat-panel';
pageHooks.trackEvent = trackEvent;
pageHooks.onSolutionLoadError = (error) => {
    addMessage('system', 'Failed to load solution. Please try again.');
};
pageHooks.onSolutionLoaded = (path) => {};
pageHooks.buildCardExtras = (top, entry) => {
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
    top.appendChild(noteBtn);
};

// ============================================================================
// Init
// ============================================================================

document.addEventListener('DOMContentLoaded', init);

async function init() {
    try {
        const configResponse = await fetch(`${API_BASE}/api/config`);
        if (configResponse.ok) {
            const config = await configResponse.json();
            posthogApiKey = config.posthog_api_key;
        }
    } catch (e) {
        console.warn('Failed to fetch PostHog config:', e);
    }

    const hasConsent = localStorage.getItem('posthog_consent') === 'true';
    const participantId = localStorage.getItem('participant_id');

    if (hasConsent && participantId) {
        initPostHog(participantId);
        await initApp();
    } else {
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
        console.warn('PostHog not available');
        return;
    }
    posthog.init(posthogApiKey, {
        api_host: 'https://us.i.posthog.com',
        autocapture: true,
        capture_pageview: true,
        session_recording: { maskAllInputs: false },
    });
    posthog.identify(participantId);
    posthog.capture('session_started', { participant_id: participantId });
}

async function initApp() {
    await fetchMetricsConfig();
    setupEventListeners();
    try { initMap(); } catch (e) { console.error('Failed to initialize map:', e); }

    try {
        await sendInitialMessage();
    } catch (e) {
        console.error('Failed to send initial message:', e);
        setProcessing(false);
    }
}

// ============================================================================
// Event Listeners
// ============================================================================

function setupEventListeners() {
    if (chatSend) {
        chatSend.addEventListener('click', () => sendMessage());
    }
    if (chatInput) {
        chatInput.addEventListener('keypress', e => {
            if (e.key === 'Enter' && !isProcessing) sendMessage();
        });
    }

    const generateFeedbackBtn = document.getElementById('generate-from-feedback-btn');
    if (generateFeedbackBtn) {
        generateFeedbackBtn.addEventListener('click', () => generateFromFeedback());
    }

    setupChartsClose();
    setupResizeHandle();
}

// ============================================================================
// Chat
// ============================================================================

async function sendInitialMessage() {
    await sendMessageToAgent('Show me the available zoning options grouped by their trade-offs.');
}

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message || isProcessing) return;

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
    setProcessing(true);
    showMapLoading(true);

    let thinkingMsg;
    try {
        thinkingMsg = addMessage('loading', 'Thinking... (this may take a moment)');
    } catch (e) {
        console.error('Failed to add thinking message:', e);
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
        if (!response.ok) throw new Error(`Chat request failed: ${response.status}`);

        const data = await response.json();
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
            if (data.text) addMessage('assistant', data.text);
            renderClusterSelector(data.clusters);
        } else if (data.response_type === 'solution_update' && data.solution_path) {
            if (data.text) addMessage('assistant', data.text);
            await loadSolution(data.solution_path);
            if (currentSolution) {
                const label = data.description || data.text?.substring(0, 50) || 'Solution';
                autoSaveSolution(currentSolution, label, data.text);
            }
        } else {
            addMessage('assistant', data.text || 'Agent returned empty response. Please try again.');
        }
    } catch (error) {
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

// ============================================================================
// Cluster Selector
// ============================================================================

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
    document.querySelectorAll('.cluster-selector').forEach(selector => {
        selector.classList.add('disabled');
        selector.querySelectorAll('.cluster-option').forEach(opt => {
            opt.onclick = null;
            opt.style.cursor = 'default';
        });
    });

    trackEvent('cluster_selected', { cluster_id: clusterId, cluster_label: clusterLabel });
    addMessage('user', `Select cluster ${clusterId}: ${clusterLabel}`);
    await sendMessageToAgent(`Select cluster ${clusterId}`);
}

// ============================================================================
// Solution Auto-Save
// ============================================================================

function autoSaveSolution(solutionData, label, agentMessage) {
    const path = solutionData.path || '';
    if (path && savedSolutions.some(s => s.path === path)) {
        const existing = savedSolutions.find(s => s.path === path);
        if (existing) {
            currentViewedIndex = existing.index;
            renderSolutionHistory();
        }
        return;
    }

    if (savedSolutions.length >= MAX_SAVED_SOLUTIONS) {
        savedSolutions.shift();
        savedSolutions.forEach((s, i) => { s.index = i + 1; });
    }

    const index = savedSolutions.length + 1;

    const categoryScores = solutionData.category_percentiles
        ? { ...solutionData.category_percentiles }
        : (() => {
            const ranks = solutionData.percentile_ranks || {};
            const scores = {};
            for (const [cat, metrics] of Object.entries(getHistoryCategories())) {
                scores[cat] = getCategoryPercentile(ranks, metrics);
            }
            return scores;
        })();

    savedSolutions.push({
        index,
        path,
        solutionData: JSON.parse(JSON.stringify(solutionData)),
        label: label || `Solution #${index}`,
        agentMessage: agentMessage || '',
        pros: '',
        cons: '',
        timestamp: new Date().toISOString(),
        categoryScores,
    });
    currentViewedIndex = index;
    renderSolutionHistory();
    updateProsConsPanel();

    trackEvent('solution_saved', {
        solution_index: index,
        label: label || `Solution #${index}`,
        solution_path: path,
    });
}

// ============================================================================
// Feedback Generation
// ============================================================================

function buildSavedSolutionsSummary() {
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

async function generateFromFeedback() {
    const hasFeedback = savedSolutions.some(s => s.pros || s.cons);
    if (!hasFeedback) {
        addMessage('system', 'Add pros/cons notes to your saved solutions first so the agent knows your preferences.');
        return;
    }

    const feedbackLines = savedSolutions
        .filter(s => s.pros || s.cons)
        .map(s => {
            const parts = [`Solution #${s.index} ("${s.label}")`];
            if (s.pros) parts.push(`Pros: ${s.pros}`);
            if (s.cons) parts.push(`Cons: ${s.cons}`);
            return parts.join(' - ');
        });

    const message = `[GENERATE_FROM_FEEDBACK] Here is ALL of my feedback across saved solutions:\n${feedbackLines.join('\n')}\n\nAnalyze every piece of feedback above, aggressively apply constraints to match my preferences, and find me a new solution.`;

    addMessage('user', 'Generate a new solution based on all my feedback');
    trackEvent('generate_from_feedback', { solution_count: savedSolutions.length, feedback_count: feedbackLines.length });
    await sendMessageToAgent(message);
}

// ============================================================================
// UI Helpers
// ============================================================================

function addMessage(type, content) {
    const div = document.createElement('div');
    div.className = `message ${type}`;

    if (type === 'assistant' && typeof marked !== 'undefined') {
        marked.setOptions({ breaks: true, gfm: true });
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
    if (chatInputArea) chatInputArea.classList.toggle('processing', processing);
}

function trackEvent(eventName, properties) {
    if (typeof posthog !== 'undefined' && posthog.capture) {
        posthog.capture(eventName, properties);
    }
}

function showLoading(show) {
    if (loadingOverlay) loadingOverlay.classList.toggle('hidden', !show);
}
