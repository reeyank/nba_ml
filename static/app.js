// NBA Stats Predictor - Multi-Model Frontend

const API_BASE = window.location.origin;

// Player headshot URL helper
function getPlayerHeadshot(playerId, size = 260) {
    return `https://cdn.nba.com/headshots/nba/latest/${size}x190/${playerId}.png`;
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    loadModelStatus();
    loadRecentGames(); // Default to recent games
});

// Load model status
async function loadModelStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();

        const statusEl = document.getElementById('model-status');
        statusEl.innerHTML = `
            <span class="status-dot"></span>
            <span class="status-text">3 Models Active</span>
        `;
    } catch (error) {
        console.error('Error loading model status:', error);
    }
}

// Load today's games
async function loadTodaysGames() {
    const container = document.getElementById('games-list');
    container.innerHTML = '<div class="loading-placeholder">Loading today\'s games...</div>';

    try {
        const response = await fetch(`${API_BASE}/api/games/today`);
        const data = await response.json();

        if (data.games.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">📅</div>
                    <p>No games scheduled for today</p>
                </div>
            `;
            return;
        }

        container.innerHTML = data.games.map(game => createGameCard(game)).join('');
    } catch (error) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">⚠️</div>
                <p>Could not load today's games</p>
            </div>
        `;
    }
}

// Load recent games from database
async function loadRecentGames() {
    const container = document.getElementById('games-list');
    container.innerHTML = '<div class="loading-placeholder">Loading recent games...</div>';

    try {
        const response = await fetch(`${API_BASE}/api/games/recent?days=14`);
        const data = await response.json();

        if (data.games.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">📅</div>
                    <p>No recent games found</p>
                </div>
            `;
            return;
        }

        container.innerHTML = data.games.map(game => createGameCard(game)).join('');
    } catch (error) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">⚠️</div>
                <p>Could not load recent games</p>
            </div>
        `;
    }
}

// Create game card
function createGameCard(game) {
    const statusClass = (game.game_status || 'scheduled').toLowerCase().replace(' ', '-');
    const homeTeamAbbr = game.home_team?.team_name?.substring(0, 3).toUpperCase() || 'HOM';
    const awayTeamAbbr = game.away_team?.team_name?.substring(0, 3).toUpperCase() || 'AWY';

    return `
        <div class="game-card" onclick="selectGame('${game.game_id}')">
            <div class="game-card-header">
                <span class="game-date">${formatDate(game.game_date)}</span>
                <span class="game-status ${statusClass}">${game.game_status || 'Scheduled'}</span>
            </div>
            <div class="game-card-body">
                <div class="team-row">
                    <div class="team-info">
                        <div class="team-abbr">${awayTeamAbbr}</div>
                        <span class="team-name">${game.away_team?.team_name || 'Away'}</span>
                    </div>
                    <span class="team-score">${game.away_team?.score || '-'}</span>
                </div>
                <div class="team-row">
                    <div class="team-info">
                        <div class="team-abbr">${homeTeamAbbr}</div>
                        <span class="team-name">${game.home_team?.team_name || 'Home'}</span>
                    </div>
                    <span class="team-score">${game.home_team?.score || '-'}</span>
                </div>
            </div>
        </div>
    `;
}

// Select game and load predictions
async function selectGame(gameId) {
    const section = document.getElementById('predictions-section');
    const container = document.getElementById('predictions-container');
    const loading = document.getElementById('predictions-loading');
    const title = document.getElementById('predictions-title');
    const subtitle = document.getElementById('predictions-subtitle');

    section.style.display = 'block';
    loading.style.display = 'flex';
    container.innerHTML = '';

    // Scroll to predictions
    section.scrollIntoView({ behavior: 'smooth' });

    try {
        const response = await fetch(`${API_BASE}/api/predictions/game/${gameId}/all-models`);
        const data = await response.json();

        loading.style.display = 'none';

        title.textContent = `${data.away_team} @ ${data.home_team}`;
        subtitle.textContent = `${formatDate(data.game_date)} | ${data.game_status} | Models: ${data.available_models.join(', ').toUpperCase()}`;

        if (!data.predictions || data.predictions.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">🏀</div>
                    <p>No predictions available for this game</p>
                </div>
            `;
            return;
        }

        // Add model legend
        let html = `
            <div class="model-legend">
                <div class="legend-item">
                    <span class="model-dot xgboost"></span>
                    <span>XGBoost</span>
                </div>
                <div class="legend-item">
                    <span class="model-dot lightgbm"></span>
                    <span>LightGBM</span>
                </div>
                <div class="legend-item">
                    <span class="model-dot lstm"></span>
                    <span>LSTM</span>
                </div>
                ${data.game_status === 'COMPLETED' ? `
                <div class="legend-item">
                    <span class="model-dot actual"></span>
                    <span>Actual</span>
                </div>
                ` : ''}
            </div>
        `;

        // Group by team
        const homePlayers = data.predictions.filter(p => p.team === 'Home');
        const awayPlayers = data.predictions.filter(p => p.team === 'Away');

        if (awayPlayers.length > 0) {
            html += createTeamSection(data.away_team, awayPlayers, data.available_models, data.game_status === 'COMPLETED');
        }

        if (homePlayers.length > 0) {
            html += createTeamSection(data.home_team, homePlayers, data.available_models, data.game_status === 'COMPLETED');
        }

        container.innerHTML = html;

    } catch (error) {
        loading.style.display = 'none';
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">⚠️</div>
                <p>Error loading predictions: ${error.message}</p>
            </div>
        `;
    }
}

// Create team section
function createTeamSection(teamName, players, availableModels, hasActuals) {
    return `
        <div class="team-section">
            <div class="team-section-header">
                <h3>${teamName}</h3>
            </div>
            ${players.map(p => createPlayerPredictionCard(p, availableModels, hasActuals)).join('')}
        </div>
    `;
}

// Create player prediction card with all models
function createPlayerPredictionCard(player, availableModels, hasActuals) {
    const headshotUrl = getPlayerHeadshot(player.player_id);
    const initial = player.player_name?.charAt(0) || 'P';

    // Build model predictions HTML
    let predictionsHtml = '';

    for (const modelType of availableModels) {
        const modelData = player.models[modelType];
        if (modelData) {
            predictionsHtml += createModelPrediction(modelType, modelData, player.actual);
        }
    }

    // Add actual stats if available
    if (hasActuals && player.actual) {
        predictionsHtml += createActualStats(player.actual, player.models, availableModels);
    }

    return `
        <div class="player-prediction-card">
            <div class="player-header">
                <div class="player-avatar">
                    <img src="${headshotUrl}" alt="${player.player_name}"
                         onerror="this.parentElement.innerHTML='<div class=\\'player-avatar-placeholder\\'>${initial}</div>'">
                </div>
                <div class="player-info">
                    <div class="player-name">${player.player_name}</div>
                    <div class="player-team">${player.team} Team</div>
                </div>
            </div>
            <div class="predictions-row">
                ${predictionsHtml}
            </div>
        </div>
    `;
}

// Create model prediction block
function createModelPrediction(modelType, data, actual) {
    const modelLabel = modelType.toUpperCase();

    return `
        <div class="model-prediction">
            <div class="model-header">
                <span class="model-dot ${modelType}"></span>
                <span class="model-name">${modelLabel}</span>
            </div>
            <div class="model-stats">
                <div class="stat-item">
                    <div class="stat-value ${modelType}">${data.points}</div>
                    <div class="stat-label">PTS</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value ${modelType}">${data.rebounds}</div>
                    <div class="stat-label">REB</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value ${modelType}">${data.assists}</div>
                    <div class="stat-label">AST</div>
                </div>
            </div>
        </div>
    `;
}

// Create actual stats block with comparison - shows per-model errors
function createActualStats(actual, models, availableModels) {
    // Calculate error for each model
    function getModelErrors() {
        let errorsHtml = '';
        for (const modelType of availableModels) {
            if (models[modelType]) {
                const ptsErr = Math.abs(models[modelType].points - actual.points).toFixed(1);
                const rebErr = Math.abs(models[modelType].rebounds - actual.rebounds).toFixed(1);
                const astErr = Math.abs(models[modelType].assists - actual.assists).toFixed(1);
                errorsHtml += `<span class="model-error ${modelType}">${modelType.toUpperCase()}: ±${ptsErr}/${rebErr}/${astErr}</span>`;
            }
        }
        return errorsHtml;
    }

    return `
        <div class="model-prediction actual-stats">
            <div class="model-header">
                <span class="model-dot actual"></span>
                <span class="model-name">FINAL STATS</span>
            </div>
            <div class="model-stats">
                <div class="stat-item">
                    <div class="stat-value actual-value">${actual.points}</div>
                    <div class="stat-label">PTS</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value actual-value">${actual.rebounds}</div>
                    <div class="stat-label">REB</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value actual-value">${actual.assists}</div>
                    <div class="stat-label">AST</div>
                </div>
            </div>
            <div class="model-errors">
                ${getModelErrors()}
            </div>
        </div>
    `;
}

// Get diff class based on error magnitude
function getDiffClass(error, statType) {
    const err = parseFloat(error);

    // Thresholds based on stat type
    if (statType === 'points') {
        if (err <= 3) return 'good';
        if (err <= 6) return 'ok';
        return 'bad';
    } else if (statType === 'rebounds') {
        if (err <= 1.5) return 'good';
        if (err <= 3) return 'ok';
        return 'bad';
    } else { // assists
        if (err <= 1) return 'good';
        if (err <= 2) return 'ok';
        return 'bad';
    }
}

// Close predictions panel
function closePredictions() {
    const section = document.getElementById('predictions-section');
    section.style.display = 'none';
}

// Format date helper
function formatDate(dateString) {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
    });
}
