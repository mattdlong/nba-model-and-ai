/**
 * NBA Model Dashboard Charts
 *
 * Chart.js initialization and data loading functions for the dashboard.
 * All functions expect a canvas element ID and a JSON data URL.
 */

// Chart.js default configuration
Chart.defaults.color = '#9ca3af';
Chart.defaults.borderColor = '#374151';
Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';

/**
 * Load and render bankroll growth chart
 * @param {string} canvasId - Canvas element ID
 * @param {string} dataUrl - URL to fetch chart data from
 */
async function loadBankrollChart(canvasId, dataUrl) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    try {
        const response = await fetch(dataUrl);
        if (!response.ok) {
            console.log('No performance data available yet');
            renderEmptyChart(canvas, 'No bankroll data available');
            return;
        }

        const data = await response.json();
        const chartData = data.charts?.bankroll;

        if (!chartData || !chartData.labels || chartData.labels.length === 0) {
            renderEmptyChart(canvas, 'No bankroll data available');
            return;
        }

        new Chart(canvas, {
            type: 'line',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return '$' + context.parsed.y.toLocaleString();
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        beginAtZero: false,
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toLocaleString();
                            }
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error loading bankroll chart:', error);
        renderEmptyChart(canvas, 'Error loading chart data');
    }
}

/**
 * Load and render ROI by month bar chart
 * @param {string} canvasId - Canvas element ID
 * @param {string} dataUrl - URL to fetch chart data from
 */
async function loadROIChart(canvasId, dataUrl) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    try {
        const response = await fetch(dataUrl);
        if (!response.ok) {
            renderEmptyChart(canvas, 'No ROI data available');
            return;
        }

        const data = await response.json();
        const chartData = data.charts?.roi_by_month;

        if (!chartData || !chartData.labels || chartData.labels.length === 0) {
            renderEmptyChart(canvas, 'No monthly ROI data available');
            return;
        }

        new Chart(canvas, {
            type: 'bar',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.parsed.y.toFixed(1) + '% ROI';
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error loading ROI chart:', error);
        renderEmptyChart(canvas, 'Error loading chart data');
    }
}

/**
 * Load and render calibration chart
 * @param {string} canvasId - Canvas element ID
 * @param {string} dataUrl - URL to fetch chart data from
 */
async function loadCalibrationChart(canvasId, dataUrl) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    try {
        const response = await fetch(dataUrl);
        if (!response.ok) {
            renderCalibrationPlaceholder(canvas);
            return;
        }

        const data = await response.json();
        const chartData = data.charts?.calibration;

        if (!chartData || !chartData.labels) {
            renderCalibrationPlaceholder(canvas);
            return;
        }

        new Chart(canvas, {
            type: 'line',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                if (context.dataset.label === 'Perfect Calibration') {
                                    return 'Perfect: ' + (context.parsed.y * 100).toFixed(0) + '%';
                                }
                                return 'Actual: ' + (context.parsed.y * 100).toFixed(1) + '%';
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Predicted Probability'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Actual Win Rate'
                        },
                        min: 0,
                        max: 1,
                        ticks: {
                            callback: function(value) {
                                return (value * 100) + '%';
                            }
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error loading calibration chart:', error);
        renderCalibrationPlaceholder(canvas);
    }
}

/**
 * Load and render win rate trend chart
 * @param {string} canvasId - Canvas element ID
 * @param {string} dataUrl - URL to fetch chart data from
 */
async function loadWinRateTrendChart(canvasId, dataUrl) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    try {
        const response = await fetch(dataUrl);
        if (!response.ok) {
            renderEmptyChart(canvas, 'No trend data available');
            return;
        }

        const data = await response.json();
        const chartData = data.charts?.win_rate_trend;

        if (!chartData || !chartData.labels || chartData.labels.length === 0) {
            renderEmptyChart(canvas, 'No trend data available');
            return;
        }

        new Chart(canvas, {
            type: 'line',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Bet Number'
                        },
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Win Rate'
                        },
                        min: 0.4,
                        max: 0.7,
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error loading win rate trend chart:', error);
        renderEmptyChart(canvas, 'Error loading chart data');
    }
}

/**
 * Load and render ROI time series chart
 * @param {string} canvasId - Canvas element ID
 * @param {string} dataUrl - URL to fetch chart data from
 */
async function loadROITimeSeriesChart(canvasId, dataUrl) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    try {
        const response = await fetch(dataUrl);
        if (!response.ok) {
            renderEmptyChart(canvas, 'No ROI time series data available');
            return;
        }

        const data = await response.json();
        const chartData = data.charts?.roi_time_series;

        if (!chartData || !chartData.labels || chartData.labels.length === 0) {
            renderEmptyChart(canvas, 'No ROI time series data available');
            return;
        }

        new Chart(canvas, {
            type: 'line',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.parsed.y.toFixed(2) + '% ROI';
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        beginAtZero: false,
                        ticks: {
                            callback: function(value) {
                                return value.toFixed(1) + '%';
                            }
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error loading ROI time series chart:', error);
        renderEmptyChart(canvas, 'Error loading chart data');
    }
}

/**
 * Load and render bet type pie chart
 * @param {string} canvasId - Canvas element ID
 * @param {string} dataUrl - URL to fetch chart data from
 */
async function loadBetTypePieChart(canvasId, dataUrl) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    try {
        const response = await fetch(dataUrl);
        if (!response.ok) {
            renderEmptyChart(canvas, 'No bet type data available');
            return;
        }

        const data = await response.json();
        const chartData = data.charts?.bet_type_breakdown;

        if (!chartData || !chartData.labels || chartData.labels.length === 0) {
            renderEmptyChart(canvas, 'No bet type data available');
            return;
        }

        new Chart(canvas, {
            type: 'pie',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const value = context.parsed;
                                const percentage = ((value / total) * 100).toFixed(1);
                                return context.label + ': ' + value + ' (' + percentage + '%)';
                            }
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error loading bet type pie chart:', error);
        renderEmptyChart(canvas, 'Error loading chart data');
    }
}

/**
 * Render empty chart placeholder
 * @param {HTMLCanvasElement} canvas - Canvas element
 * @param {string} message - Message to display
 */
function renderEmptyChart(canvas, message) {
    const ctx = canvas.getContext('2d');
    ctx.font = '14px -apple-system, sans-serif';
    ctx.fillStyle = '#6b7280';
    ctx.textAlign = 'center';
    ctx.fillText(message, canvas.width / 2, canvas.height / 2);
}

/**
 * Render calibration chart placeholder with perfect line
 * @param {HTMLCanvasElement} canvas - Canvas element
 */
function renderCalibrationPlaceholder(canvas) {
    // Show perfect calibration line as placeholder
    const bins = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'];
    const perfectLine = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

    new Chart(canvas, {
        type: 'line',
        data: {
            labels: bins,
            datasets: [{
                label: 'Perfect Calibration',
                data: perfectLine,
                borderColor: 'rgb(107, 114, 128)',
                borderDash: [5, 5],
                pointRadius: 0,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                title: {
                    display: true,
                    text: 'No calibration data - showing perfect line',
                    color: '#6b7280'
                }
            },
            scales: {
                y: {
                    min: 0,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            return (value * 100) + '%';
                        }
                    }
                }
            }
        }
    });
}

/**
 * Fetch and display today's signals summary
 * @param {string} containerId - Container element ID
 * @param {string} dataUrl - URL to fetch signals from
 */
async function loadSignalsSummary(containerId, dataUrl) {
    const container = document.getElementById(containerId);
    if (!container) return;

    try {
        const response = await fetch(dataUrl);
        if (!response.ok) {
            container.innerHTML = '<p class="no-data">No signals available</p>';
            return;
        }

        const data = await response.json();
        const signals = data.signals || [];

        if (signals.length === 0) {
            container.innerHTML = '<p class="no-data">No signals for today</p>';
            return;
        }

        // Render signals
        const html = signals.slice(0, 5).map(signal => `
            <div class="signal-item">
                <span class="matchup">${signal.matchup}</span>
                <span class="bet">${signal.bet_type} ${signal.side}</span>
                <span class="edge ${signal.edge > 0 ? 'positive' : 'negative'}">
                    ${(signal.edge * 100).toFixed(1)}%
                </span>
            </div>
        `).join('');

        container.innerHTML = html;
    } catch (error) {
        console.error('Error loading signals:', error);
        container.innerHTML = '<p class="no-data">Error loading signals</p>';
    }
}

// Initialize charts when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Auto-initialize charts if data attributes are present
    document.querySelectorAll('[data-chart]').forEach(canvas => {
        const chartType = canvas.dataset.chart;
        const dataUrl = canvas.dataset.url || 'api/performance.json';

        switch (chartType) {
            case 'bankroll':
                loadBankrollChart(canvas.id, dataUrl);
                break;
            case 'roi':
                loadROIChart(canvas.id, dataUrl);
                break;
            case 'roi-time-series':
                loadROITimeSeriesChart(canvas.id, dataUrl);
                break;
            case 'calibration':
                loadCalibrationChart(canvas.id, dataUrl);
                break;
            case 'winrate':
                loadWinRateTrendChart(canvas.id, dataUrl);
                break;
            case 'bet-type-pie':
                loadBetTypePieChart(canvas.id, dataUrl);
                break;
        }
    });
});
