// Nifty AI Dashboard JavaScript
class NiftyDashboard {
    constructor() {
        this.data = {
            "nifty_data": [
                {
                    "Date": "2023-10-23",
                    "Open": 61850.2,
                    "High": 62100.5,
                    "Low": 61750.1,
                    "Close": 61902.1,
                    "Volume": 1050000,
                    "RSI": 50.8,
                    "MACD": 142.6,
                    "MACD_signal": 138.2,
                    "SMA_5": 62150.3,
                    "SMA_20": 61800.5,
                    "SMA_50": 61500.2,
                    "AI_Score": -0.4,
                    "AI_Signal": "Bearish",
                    "BB_upper": 62500.0,
                    "BB_lower": 61200.0
                },
                {
                    "Date": "2023-10-24",
                    "Open": 61902.1,
                    "High": 62850.0,
                    "Low": 61850.0,
                    "Close": 62768.3,
                    "Volume": 1120000,
                    "RSI": 47.8,
                    "MACD": 180.8,
                    "MACD_signal": 145.5,
                    "SMA_5": 62250.8,
                    "SMA_20": 61850.2,
                    "SMA_50": 61600.1,
                    "AI_Score": -0.21,
                    "AI_Signal": "Neutral",
                    "BB_upper": 62600.0,
                    "BB_lower": 61100.0
                },
                {
                    "Date": "2023-10-25",
                    "Open": 62768.3,
                    "High": 63200.0,
                    "Low": 62650.0,
                    "Close": 63089.0,
                    "Volume": 980000,
                    "RSI": 55.3,
                    "MACD": 234.3,
                    "MACD_signal": 152.1,
                    "SMA_5": 62400.5,
                    "SMA_20": 61900.8,
                    "SMA_50": 61750.3,
                    "AI_Score": 0.37,
                    "AI_Signal": "Bullish",
                    "BB_upper": 62800.0,
                    "BB_lower": 61000.0
                },
                {
                    "Date": "2023-10-26",
                    "Open": 63089.0,
                    "High": 64000.0,
                    "Low": 62900.0,
                    "Close": 63889.8,
                    "Volume": 1180000,
                    "RSI": 48.8,
                    "MACD": 337.4,
                    "MACD_signal": 165.8,
                    "SMA_5": 62650.2,
                    "SMA_20": 62000.1,
                    "SMA_50": 61850.0,
                    "AI_Score": 0.34,
                    "AI_Signal": "Bullish",
                    "BB_upper": 63000.0,
                    "BB_lower": 60900.0
                },
                {
                    "Date": "2023-10-27",
                    "Open": 63889.8,
                    "High": 65377.9,
                    "Low": 63800.0,
                    "Close": 64525.1,
                    "Volume": 1050000,
                    "RSI": 66.1,
                    "MACD": 465.0,
                    "MACD_signal": 178.5,
                    "SMA_5": 63200.0,
                    "SMA_20": 62150.5,
                    "SMA_50": 62000.0,
                    "AI_Score": 0.18,
                    "AI_Signal": "Neutral",
                    "BB_upper": 63200.0,
                    "BB_lower": 60800.0
                }
            ],
            "market_insights": {
                "current_signal": "Neutral",
                "signal_strength": 18.0,
                "trend_direction": "Upward",
                "rsi_condition": "Neutral",
                "volume_analysis": "Normal",
                "volatility_level": "High",
                "support_level": 59434.4,
                "resistance_level": 65377.9,
                "confidence_score": 54.5
            },
            "ai_features": {
                "pattern_recognition": [
                    "Moving Average Crossover: SMA 5 above SMA 20 (Bullish)",
                    "MACD: Above signal line (Positive momentum)",
                    "Bollinger Bands: Price near middle band (Neutral)",
                    "Volume: Normal trading activity"
                ],
                "risk_assessment": "Medium Risk",
                "recommendation": "Hold positions with tight stop losses. Monitor for breakout above 65400 resistance.",
                "key_levels": {
                    "strong_support": 61200,
                    "weak_support": 62800,
                    "weak_resistance": 64800,
                    "strong_resistance": 65400
                }
            },
            "historical_signals": [
                {"date": "2023-10-20", "signal": "Neutral", "price": 61344, "accuracy": "Correct"},
                {"date": "2023-10-21", "signal": "Neutral", "price": 62157, "accuracy": "Correct"},
                {"date": "2023-10-22", "signal": "Bearish", "price": 62521, "accuracy": "Correct"},
                {"date": "2023-10-23", "signal": "Bearish", "price": 61902, "accuracy": "Correct"},
                {"date": "2023-10-24", "signal": "Neutral", "price": 62768, "accuracy": "Correct"},
                {"date": "2023-10-25", "signal": "Bullish", "price": 63089, "accuracy": "Correct"},
                {"date": "2023-10-26", "signal": "Bullish", "price": 63890, "accuracy": "Correct"},
                {"date": "2023-10-27", "signal": "Neutral", "price": 64525, "accuracy": "Pending"}
            ]
        };

        this.charts = {};
        this.chartOptions = {
            showMA: true,
            showBB: true,
            showVolume: true
        };
        
        this.init();
    }

    init() {
        this.updateDateTime();
        this.updateMetrics();
        this.initCharts();
        this.updateIndicatorsTable();
        this.updateInsights();
        this.updateSignalHistory();
        this.setupEventListeners();
        
        // Update every 30 seconds to simulate live data
        setInterval(() => {
            this.simulateDataUpdate();
        }, 30000);
        
        setInterval(() => {
            this.updateDateTime();
        }, 1000);
    }

    updateDateTime() {
        const now = new Date();
        const options = { 
            weekday: 'long', 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric',
            hour: 'numeric',
            minute: '2-digit',
            hour12: true
        };
        document.getElementById('currentDateTime').textContent = now.toLocaleDateString('en-US', options);
    }

    updateMetrics() {
        const latest = this.data.nifty_data[this.data.nifty_data.length - 1];
        const insights = this.data.market_insights;
        
        // Update AI Signal
        const signalElement = document.getElementById('aiSignal');
        const signalCard = signalElement.closest('.metric-card');
        signalElement.textContent = latest.AI_Signal;
        signalElement.className = `signal-badge ${latest.AI_Signal.toLowerCase()}`;
        
        // Update Signal Strength
        document.getElementById('signalStrength').textContent = Math.abs(latest.AI_Score * 100).toFixed(0);
        
        // Update Current Price
        const priceElement = document.getElementById('currentPrice');
        priceElement.textContent = this.formatNumber(latest.Close);
        
        // Calculate price change
        const previous = this.data.nifty_data[this.data.nifty_data.length - 2];
        const change = latest.Close - previous.Close;
        const changePercent = (change / previous.Close * 100);
        const priceChangeElement = document.getElementById('priceChange');
        priceChangeElement.textContent = `${change >= 0 ? '+' : ''}${this.formatNumber(change)} (${changePercent >= 0 ? '+' : ''}${changePercent.toFixed(1)}%)`;
        priceChangeElement.className = `metric-change ${change >= 0 ? 'positive' : 'negative'}`;
        
        // Update Trend Direction
        document.getElementById('trendDirection').textContent = insights.trend_direction;
        const trendArrow = document.querySelector('.trend-arrow');
        trendArrow.textContent = insights.trend_direction === 'Upward' ? '↗' : '↘';
        trendArrow.className = `trend-arrow ${insights.trend_direction === 'Upward' ? 'up' : 'down'}`;
        
        // Update RSI
        document.getElementById('rsiValue').textContent = latest.RSI.toFixed(1);
        const rsiCondition = document.getElementById('rsiCondition');
        if (latest.RSI > 70) {
            rsiCondition.textContent = 'Overbought';
            rsiCondition.className = 'metric-change negative';
        } else if (latest.RSI < 30) {
            rsiCondition.textContent = 'Oversold';
            rsiCondition.className = 'metric-change positive';
        } else {
            rsiCondition.textContent = 'Neutral';
            rsiCondition.className = 'metric-change neutral';
        }
        
        // Update Confidence Score
        document.getElementById('confidenceScore').textContent = insights.confidence_score.toFixed(1);
    }

    initCharts() {
        this.createPriceChart();
        this.createAIScoreChart();
    }

    createPriceChart() {
        const ctx = document.getElementById('priceChart').getContext('2d');
        
        const labels = this.data.nifty_data.map(d => d.Date);
        const prices = this.data.nifty_data.map(d => ({
            x: d.Date,
            o: d.Open,
            h: d.High,
            l: d.Low,
            c: d.Close
        }));
        const sma5 = this.data.nifty_data.map(d => d.SMA_5);
        const sma20 = this.data.nifty_data.map(d => d.SMA_20);
        const sma50 = this.data.nifty_data.map(d => d.SMA_50);
        const bbUpper = this.data.nifty_data.map(d => d.BB_upper);
        const bbLower = this.data.nifty_data.map(d => d.BB_lower);

        this.charts.priceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Nifty 50 Price',
                        data: this.data.nifty_data.map(d => d.Close),
                        borderColor: '#1FB8CD',
                        backgroundColor: 'rgba(31, 184, 205, 0.1)',
                        borderWidth: 2,
                        tension: 0.1,
                        pointRadius: 4,
                        pointHoverRadius: 6
                    },
                    {
                        label: 'SMA 5',
                        data: sma5,
                        borderColor: '#FFC185',
                        borderWidth: 1,
                        tension: 0.1,
                        pointRadius: 0,
                        hidden: !this.chartOptions.showMA
                    },
                    {
                        label: 'SMA 20',
                        data: sma20,
                        borderColor: '#B4413C',
                        borderWidth: 1,
                        tension: 0.1,
                        pointRadius: 0,
                        hidden: !this.chartOptions.showMA
                    },
                    {
                        label: 'SMA 50',
                        data: sma50,
                        borderColor: '#5D878F',
                        borderWidth: 1,
                        tension: 0.1,
                        pointRadius: 0,
                        hidden: !this.chartOptions.showMA
                    },
                    {
                        label: 'Bollinger Upper',
                        data: bbUpper,
                        borderColor: '#ECEBD5',
                        borderWidth: 1,
                        borderDash: [5, 5],
                        tension: 0.1,
                        pointRadius: 0,
                        hidden: !this.chartOptions.showBB
                    },
                    {
                        label: 'Bollinger Lower',
                        data: bbLower,
                        borderColor: '#ECEBD5',
                        borderWidth: 1,
                        borderDash: [5, 5],
                        tension: 0.1,
                        pointRadius: 0,
                        hidden: !this.chartOptions.showBB
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: 'var(--color-text)',
                            font: {
                                size: 12
                            }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        borderColor: '#1FB8CD',
                        borderWidth: 1,
                        callbacks: {
                            title: function(context) {
                                return context[0].label;
                            },
                            label: function(context) {
                                return context.dataset.label + ': ₹' + context.parsed.y.toLocaleString();
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: 'var(--color-text-secondary)'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    },
                    y: {
                        ticks: {
                            color: 'var(--color-text-secondary)',
                            callback: function(value) {
                                return '₹' + value.toLocaleString();
                            }
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                }
            }
        });
    }

    createAIScoreChart() {
        const ctx = document.getElementById('aiScoreChart').getContext('2d');
        
        const labels = this.data.nifty_data.map(d => d.Date);
        const aiScores = this.data.nifty_data.map(d => d.AI_Score * 100);

        this.charts.aiScoreChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'AI Signal Strength',
                    data: aiScores,
                    backgroundColor: aiScores.map(score => 
                        score > 30 ? '#1FB8CD' : 
                        score < -30 ? '#DB4545' : '#D2BA4C'
                    ),
                    borderColor: aiScores.map(score => 
                        score > 30 ? '#1FB8CD' : 
                        score < -30 ? '#DB4545' : '#D2BA4C'
                    ),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        callbacks: {
                            label: function(context) {
                                const value = context.parsed.y;
                                const signal = value > 30 ? 'Bullish' : value < -30 ? 'Bearish' : 'Neutral';
                                return `${signal}: ${Math.abs(value).toFixed(1)}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: 'var(--color-text-secondary)',
                            font: {
                                size: 10
                            }
                        },
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        beginAtZero: true,
                        min: -100,
                        max: 100,
                        ticks: {
                            color: 'var(--color-text-secondary)',
                            font: {
                                size: 10
                            },
                            callback: function(value) {
                                return value + '%';
                            }
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                }
            }
        });
    }

    updateIndicatorsTable() {
        const latest = this.data.nifty_data[this.data.nifty_data.length - 1];
        
        document.getElementById('rsiTableValue').textContent = latest.RSI.toFixed(1);
        document.getElementById('macdTableValue').textContent = latest.MACD.toFixed(1);
        document.getElementById('sma5TableValue').textContent = this.formatNumber(latest.SMA_5);
        document.getElementById('sma20TableValue').textContent = this.formatNumber(latest.SMA_20);
        document.getElementById('volumeTableValue').textContent = latest.Volume.toLocaleString();
        document.getElementById('supportResistanceValue').textContent = 
            `${this.formatNumber(this.data.ai_features.key_levels.strong_support)} / ${this.formatNumber(this.data.ai_features.key_levels.strong_resistance)}`;
    }

    updateInsights() {
        const insights = this.data.market_insights;
        const aiFeatures = this.data.ai_features;
        
        document.getElementById('marketCondition').textContent = 
            `The market is showing ${insights.current_signal.toLowerCase()} sentiment with moderate ${insights.trend_direction.toLowerCase()} undertones. Recent price action suggests consolidation around current levels with potential for ${insights.trend_direction.toLowerCase()} movement.`;
        
        // Update pattern recognition
        const patternList = document.getElementById('patternList');
        patternList.innerHTML = '';
        aiFeatures.pattern_recognition.forEach(pattern => {
            const li = document.createElement('li');
            li.textContent = pattern;
            patternList.appendChild(li);
        });
        
        // Update risk assessment
        document.getElementById('riskLevel').textContent = aiFeatures.risk_assessment;
        document.getElementById('riskLevel').className = `risk-level ${aiFeatures.risk_assessment.toLowerCase().split(' ')[0]}`;
        document.getElementById('riskDescription').textContent = 
            `Current market conditions present ${aiFeatures.risk_assessment.toLowerCase()}. Volatility is ${insights.volatility_level.toLowerCase()}, requiring careful position sizing and risk management.`;
        
        // Update trading recommendation
        document.getElementById('tradingRecommendation').textContent = aiFeatures.recommendation;
    }

    updateSignalHistory() {
        const tbody = document.getElementById('historyTableBody');
        tbody.innerHTML = '';
        
        this.data.historical_signals.slice().reverse().forEach(signal => {
            const row = tbody.insertRow();
            
            // Format date
            const date = new Date(signal.date);
            const formattedDate = date.toLocaleDateString('en-US', { 
                month: 'short', 
                day: 'numeric' 
            });
            
            row.innerHTML = `
                <td>${formattedDate}</td>
                <td><span class="signal-history-badge ${signal.signal.toLowerCase()}">${signal.signal}</span></td>
                <td>₹${signal.price.toLocaleString()}</td>
                <td><span class="accuracy-badge ${signal.accuracy.toLowerCase()}">${signal.accuracy}</span></td>
            `;
        });
    }

    setupEventListeners() {
        // Chart toggle buttons
        document.getElementById('toggleMA').addEventListener('click', () => {
            this.chartOptions.showMA = !this.chartOptions.showMA;
            this.toggleDatasetVisibility('priceChart', [1, 2, 3], this.chartOptions.showMA);
        });
        
        document.getElementById('toggleBB').addEventListener('click', () => {
            this.chartOptions.showBB = !this.chartOptions.showBB;
            this.toggleDatasetVisibility('priceChart', [4, 5], this.chartOptions.showBB);
        });
        
        document.getElementById('toggleVolume').addEventListener('click', () => {
            this.chartOptions.showVolume = !this.chartOptions.showVolume;
            // Volume chart logic would go here if implemented
        });
    }

    toggleDatasetVisibility(chartName, datasetIndices, show) {
        const chart = this.charts[chartName];
        datasetIndices.forEach(index => {
            chart.data.datasets[index].hidden = !show;
        });
        chart.update();
    }

    simulateDataUpdate() {
        // Simulate small price movements for live feel
        const latest = this.data.nifty_data[this.data.nifty_data.length - 1];
        const randomChange = (Math.random() - 0.5) * 100; // ±50 points
        
        // Don't actually modify data, just add visual feedback
        const signalCard = document.querySelector('.signal-card');
        signalCard.classList.add('updated');
        setTimeout(() => {
            signalCard.classList.remove('updated');
        }, 2000);
    }

    formatNumber(num) {
        return new Intl.NumberFormat('en-IN', {
            minimumFractionDigits: 0,
            maximumFractionDigits: 1
        }).format(num);
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new NiftyDashboard();
});

// Handle theme switching if needed
function detectColorScheme() {
    const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    return isDark ? 'dark' : 'light';
}

// Update charts on theme change
window.matchMedia('(prefers-color-scheme: dark)').addListener(() => {
    // Could rebuild charts here if needed for theme changes
});