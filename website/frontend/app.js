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
const versionTabsEl = document.getElementById('version-tabs');
const generateBtn = document.getElementById('generate-btn');

// Pre-version chat messages (cluster selection, welcome) stored here until first version created
let preVersionMessages = [];

// ============================================================================
// Page Hooks (configure shared.js behavior for user page)
// ============================================================================

pageHooks.rightPanelSelector = '#chat-panel';
pageHooks.trackEvent = trackEvent;
pageHooks.onSolutionLoadError = () => {
    addMessage('system', 'Failed to load solution. Please try again.');
};
pageHooks.onSolutionLoaded = () => {};
pageHooks.buildCardExtras = () => {};

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
    if (chatSend) chatSend.addEventListener('click', () => sendMessage());
    if (chatInput) {
        chatInput.addEventListener('keypress', e => {
            if (e.key === 'Enter' && !isProcessing) sendMessage();
        });
    }
    if (generateBtn) generateBtn.addEventListener('click', () => generateNewSolution());

    setupChartsClose();
    setupResizeHandle();
}

// ============================================================================
// Version Tabs
// ============================================================================

function renderVersionTabs() {
    if (!versionTabsEl) return;

    if (versions.length === 0) {
        versionTabsEl.classList.add('hidden');
        return;
    }

    versionTabsEl.classList.remove('hidden');
    versionTabsEl.innerHTML = '';

    versions.forEach(v => {
        const tab = document.createElement('button');
        tab.className = 'version-tab' + (v.id === currentVersionId ? ' active' : '');
        tab.addEventListener('click', () => switchToVersion(v.id));

        // Version label
        const label = document.createElement('span');
        label.className = 'version-tab-label';
        label.textContent = `v${v.id}`;
        tab.appendChild(label);

        // Category percentile badges
        const scores = v.categoryScores || {};
        if (Object.keys(scores).length > 0) {
            const badges = document.createElement('div');
            badges.className = 'version-tab-badges';
            for (const [catShort, pct] of Object.entries(scores)) {
                if (pct == null) continue;
                const rounded = Math.round(pct);
                const ranking = getPercentileRanking(rounded);
                const badge = document.createElement('span');
                badge.className = `version-badge ${ranking}`;
                badge.textContent = `${catShort}: ${rounded}%`;
                badges.appendChild(badge);
            }
            tab.appendChild(badges);
        }

        // Description
        if (v.label) {
            const desc = document.createElement('span');
            desc.className = 'version-tab-desc';
            desc.textContent = v.label;
            tab.appendChild(desc);
        }

        versionTabsEl.appendChild(tab);
    });
}

function switchToVersion(versionId) {
    if (versionId === currentVersionId) return;

    currentVersionId = versionId;
    viewVersion(versionId);
    renderVersionTabs();
    renderChatForVersion(versionId);
    updateInputState();
}

function updateInputState() {
    const isLatest = isLatestVersion(currentVersionId);
    if (chatInput) chatInput.disabled = !isLatest;
    if (chatSend) chatSend.disabled = !isLatest;
    if (chatInputArea) chatInputArea.classList.toggle('read-only', !isLatest);
}

// ============================================================================
// Per-Version Chat Rendering
// ============================================================================

function renderChatForVersion(versionId) {
    const entry = versions.find(v => v.id === versionId);
    if (!entry) return;

    chatMessages.innerHTML = '';
    entry.chatMessages.forEach(msg => {
        const div = document.createElement('div');
        div.className = `message ${msg.type}`;
        if (msg.type === 'assistant' && typeof marked !== 'undefined') {
            marked.setOptions({ breaks: true, gfm: true });
            div.innerHTML = marked.parse(msg.content);
        } else {
            div.textContent = msg.content;
        }
        chatMessages.appendChild(div);
    });

    chatMessages.scrollTop = chatMessages.scrollHeight;
}

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

    // Store in current version's chat history (skip loading messages)
    if (type !== 'loading') {
        const entry = versions.find(v => v.id === currentVersionId);
        if (entry) {
            entry.chatMessages.push({ type, content });
        } else {
            preVersionMessages.push({ type, content });
        }
    }

    return div;
}

// ============================================================================
// Chat
// ============================================================================

async function sendInitialMessage() {
    setProcessing(true);
    showMapLoading(true);
    try {
        const url = sessionId
            ? `${API_BASE}/api/initial-clusters?session_id=${sessionId}`
            : `${API_BASE}/api/initial-clusters`;
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Failed to fetch clusters: ${response.status}`);
        const data = await response.json();
        sessionId = data.session_id;

        if (data.clusters && data.clusters.length > 0) {
            if (data.text) addMessage('assistant', data.text);
            renderClusterSelector(data.clusters);
        } else {
            addMessage('assistant', 'No clusters available. You can start chatting directly.');
            setProcessing(false);
        }
    } catch (e) {
        console.error('Failed to fetch initial clusters:', e);
        addMessage('system', 'Failed to load initial clusters. Please refresh.');
        setProcessing(false);
    } finally {
        showMapLoading(false);
    }
}

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message || isProcessing) return;
    if (!isLatestVersion(currentVersionId)) return;

    addMessage('user', message);
    chatInput.value = '';

    trackEvent('chat_message_sent', {
        message_text: message,
        message_length: message.length,
        session_id: sessionId,
    });

    const data = await sendMessageToAgent(message, 'feedback');
    if (data && data.text) {
        addMessage('assistant', data.text);
    }
}

async function sendMessageToAgent(message, mode = 'feedback') {
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
                mode,
                current_solution_index: currentVersionId,
                saved_solutions: buildVersionsSummary(),
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
            has_solution: !!data.solution_path,
            mode,
            session_id: sessionId,
        });

        return data;
    } catch (error) {
        if (thinkingMsg) thinkingMsg.remove();
        if (error.name === 'AbortError') {
            addMessage('assistant', 'Request timed out. Please try again.');
        } else {
            addMessage('assistant', `Error: ${error.message}. Please try again.`);
        }
        return null;
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
    });

    trackEvent('cluster_selected', { cluster_id: clusterId, cluster_label: clusterLabel });
    setProcessing(true);
    showMapLoading(true);

    try {
        const response = await fetch(`${API_BASE}/api/select-cluster`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, cluster_id: clusterId }),
        });
        if (!response.ok) throw new Error(`Select cluster failed: ${response.status}`);
        const data = await response.json();

        if (data.solution_path) {
            await loadSolution(data.solution_path);
            if (currentSolution) {
                createVersion(
                    currentSolution,
                    data.solution_path,
                    data.description || clusterLabel,
                    data.text,
                );
            }

            // Ask the agent to prompt for initial feedback
            const feedbackPrompt = await sendMessageToAgent(
                `I just selected the "${clusterLabel}" cluster. Ask me what I think of this map.`,
                'feedback',
            );
            if (feedbackPrompt && feedbackPrompt.text) {
                addMessage('assistant', feedbackPrompt.text);
            }
        }
    } catch (error) {
        addMessage('system', `Error selecting cluster: ${error.message}`);
    } finally {
        setProcessing(false);
        showMapLoading(false);
    }
}

// ============================================================================
// Version Creation
// ============================================================================

function createVersion(solutionData, solutionPath, label, agentMessage) {
    if (versions.length >= MAX_VERSIONS) {
        versions.shift();
    }

    const id = versions.length + 1;
    const categoryScores = solutionData.category_percentiles
        ? { ...solutionData.category_percentiles }
        : {};

    // Move pre-version messages (cluster selection flow) into the first version's chat
    const initialMessages = preVersionMessages.splice(0, preVersionMessages.length);

    const entry = {
        id,
        label: label || `Map #${id}`,
        solutionData: JSON.parse(JSON.stringify(solutionData)),
        solutionPath: solutionPath || '',
        chatMessages: [...initialMessages],
        categoryScores,
        timestamp: new Date().toISOString(),
    };

    if (agentMessage) {
        entry.chatMessages.push({ type: 'assistant', content: agentMessage });
    }

    versions.push(entry);
    currentVersionId = id;

    renderVersionTabs();
    renderChatForVersion(id);
    updateInputState();

    trackEvent('version_created', {
        version_id: id,
        label: entry.label,
        solution_path: solutionPath,
    });
}

// ============================================================================
// Generate New Solution
// ============================================================================

function buildVersionsSummary() {
    return versions.map(v => {
        const metrics = v.solutionData ? v.solutionData.metrics || {} : {};
        return {
            index: v.id,
            label: v.label,
            key_metrics: {
                frl: metrics.frl_dissim,
                diversity: metrics.theil_index,
                proximity: metrics.avg_closest_zone_school_distance,
                programs: metrics.avg_total_programs_per_zone,
            },
        };
    });
}

async function generateNewSolution() {
    if (isProcessing) return;

    // Flush any unsent text in the input box as a user message first
    const pending = chatInput ? chatInput.value.trim() : '';
    if (pending) {
        addMessage('user', pending);
        chatInput.value = '';
        // Send as feedback so the agent can save_feedback before generating
        await sendMessageToAgent(pending, 'feedback');
    }

    trackEvent('generate_new_solution', {
        version_count: versions.length,
        session_id: sessionId,
    });

    const data = await sendMessageToAgent(
        '[GENERATE_NEW_MAP] Generate a new map using all accumulated feedback.',
        'generate',
    );

    if (data && data.response_type === 'solution_update' && data.solution_path) {
        await loadSolution(data.solution_path);
        if (currentSolution) {
            createVersion(
                currentSolution,
                data.solution_path,
                data.description || 'Feedback-based map',
                data.text,
            );
        }
    } else if (data && data.text) {
        addMessage('assistant', data.text);
    }
}

// ============================================================================
// UI Helpers
// ============================================================================

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
