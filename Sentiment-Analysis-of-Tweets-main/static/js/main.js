document.addEventListener("DOMContentLoaded", () => {
    // Dom Elements
    const tabButtons = document.querySelectorAll(".tab-btn");
    const viewPanels = document.querySelectorAll(".view-panel");
    const tweetForm = document.getElementById("tweetForm");
    const tweetTextarea = document.getElementById("tweetText");
    const charCount = document.getElementById("charCount");
    
    // Result elements
    const resultPanel = document.getElementById("resultPanel");
    const resultPlaceholder = document.getElementById("resultPlaceholder");
    const resultEmoji = document.getElementById("resultEmoji");
    const resultSentiment = document.getElementById("resultSentiment");
    const resultConfidence = document.getElementById("resultConfidence");
    const cleanTextCode = document.getElementById("cleanTextCode");
    
    const posMeter = document.getElementById("posMeter");
    const posPct = document.getElementById("posPct");
    const neuMeter = document.getElementById("neuMeter");
    const neuPct = document.getElementById("neuPct");
    const negMeter = document.getElementById("negMeter");
    const negPct = document.getElementById("negPct");
    
    // File upload elements
    const uploadForm = document.getElementById("uploadForm");
    const uploadZone = document.getElementById("uploadZone");
    const csvFileInput = document.getElementById("csvFile");
    const batchResultSection = document.getElementById("batchResultSection");
    const batchTableBody = document.getElementById("batchTableBody");
    const downloadBtn = document.getElementById("downloadBtn");
    
    // History elements
    const historyTableBody = document.getElementById("historyTableBody");
    const clearHistoryBtn = document.getElementById("clearHistoryBtn");
    const filterSentiment = document.getElementById("filterSentiment");
    const searchHistory = document.getElementById("searchHistory");
    
    // Chart instances
    let donutChartInstance = null;
    let comparisonChartInstance = null;
    
    // Global data references
    let lastBatchFilename = null;
    let modelMetadata = null;
    
    // Textarea Character Counter
    tweetTextarea.addEventListener("input", () => {
        const count = tweetTextarea.value.length;
        charCount.textContent = `${count} characters`;
    });

    // Tab Switching
    tabButtons.forEach(button => {
        button.addEventListener("click", () => {
            const targetTab = button.dataset.tab;
            
            // Toggle active classes on buttons
            tabButtons.forEach(btn => btn.classList.remove("active"));
            button.classList.add("active");
            
            // Toggle active classes on panels
            viewPanels.forEach(panel => {
                panel.classList.remove("active");
                if (panel.id === `${targetTab}Panel`) {
                    panel.classList.add("active");
                }
            });
            
            // Extra actions when entering specific tabs
            if (targetTab === "dashboard") {
                loadDashboardData();
            } else if (targetTab === "history") {
                loadHistory();
            }
        });
    });

    // Initialize Page
    loadOverviewStats();

    // 1. Single Tweet Sentiment Analysis
    tweetForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const text = tweetTextarea.value.trim();
        if (!text) return;
        
        // Show loading state in button
        const submitBtn = tweetForm.querySelector("button[type='submit']");
        const originalBtnText = submitBtn.innerHTML;
        submitBtn.disabled = true;
        submitBtn.innerHTML = `<div class="loader-spinner" style="width: 20px; height: 20px; margin: 0; border-width: 2px;"></div> Analyzing...`;
        
        const formData = new FormData();
        formData.append("tweet", text);
        
        fetch("/analyze", {
            method: "POST",
            body: formData,
            headers: {
                "X-Requested-With": "XMLHttpRequest"
            }
        })
        .then(res => res.json())
        .then(data => {
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalBtnText;
            
            if (data.error) {
                alert(data.error);
                return;
            }
            
            // Display Results
            resultPlaceholder.style.display = "none";
            resultPanel.style.display = "flex";
            resultPanel.classList.add("has-result");
            
            // Set Emoji & Label
            resultSentiment.textContent = data.sentiment;
            resultConfidence.textContent = `Confidence Score: ${data.confidence}%`;
            cleanTextCode.textContent = data.clean_text || "(empty)";
            
            // Setup Sentiment Emoticon
            resultEmoji.className = "sentiment-emoji";
            if (data.sentiment === "Positive") {
                resultEmoji.textContent = "😊";
                resultEmoji.classList.add("positive");
                resultSentiment.style.color = "var(--color-positive)";
            } else if (data.sentiment === "Neutral") {
                resultEmoji.textContent = "😐";
                resultEmoji.classList.add("neutral");
                resultSentiment.style.color = "var(--color-neutral)";
            } else {
                resultEmoji.textContent = "😞";
                resultEmoji.classList.add("negative");
                resultSentiment.style.color = "var(--color-negative)";
            }
            
            // Update Confidence Meters
            const probs = data.probabilities;
            const posVal = (probs.Positive * 100).toFixed(1);
            const neuVal = (probs.Neutral * 100).toFixed(1);
            const negVal = (probs.Negative * 100).toFixed(1);
            
            posMeter.style.width = `${posVal}%`;
            posPct.textContent = `${posVal}%`;
            
            neuMeter.style.width = `${neuVal}%`;
            neuPct.textContent = `${neuVal}%`;
            
            negMeter.style.width = `${negVal}%`;
            negPct.textContent = `${negVal}%`;
            
            // Refresh counts
            loadOverviewStats();
        })
        .catch(err => {
            console.error("Error analyzing tweet:", err);
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalBtnText;
            alert("Analysis failed. Check backend console logs.");
        });
    });

    // 2. CSV File Batch Upload Logic
    // Drag & Drop visual highlights
    ["dragenter", "dragover"].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            uploadZone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            uploadZone.classList.remove("dragover");
        }, false);
    });

    uploadZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            csvFileInput.files = files;
            handleCSVSelection(files[0]);
        }
    });

    uploadZone.addEventListener("click", () => {
        csvFileInput.click();
    });

    csvFileInput.addEventListener("change", () => {
        if (csvFileInput.files.length) {
            handleCSVSelection(csvFileInput.files[0]);
        }
    });

    function handleCSVSelection(file) {
        uploadZone.querySelector("p").textContent = `Selected: ${file.name}`;
        uploadZone.querySelector("span").textContent = `${(file.size / 1024).toFixed(2)} KB - Click or drag to change`;
    }

    uploadForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const file = csvFileInput.files[0];
        if (!file) {
            alert("Please select a CSV file first.");
            return;
        }
        
        // Show loader spinner
        const submitBtn = uploadForm.querySelector("button[type='submit']");
        const originalBtnText = submitBtn.innerHTML;
        submitBtn.disabled = true;
        submitBtn.innerHTML = `<div class="loader-spinner" style="width: 20px; height: 20px; margin: 0; border-width: 2px;"></div> Processing File...`;
        
        batchResultSection.innerHTML = `<div class="loader-spinner"></div><p style="text-align: center; color: var(--text-secondary);">Analyzing tweets in CSV. Please wait...</p>`;
        batchResultSection.style.display = "block";
        
        const formData = new FormData();
        formData.append("file", file);
        
        fetch("/upload", {
            method: "POST",
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalBtnText;
            
            if (data.error) {
                batchResultSection.innerHTML = `<div class="empty-state"><i class="fas fa-exclamation-triangle"></i><p>${data.error}</p></div>`;
                return;
            }
            
            lastBatchFilename = data.output_file;
            renderBatchResults(data.predictions);
            loadOverviewStats();
        })
        .catch(err => {
            console.error("Error uploading CSV:", err);
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalBtnText;
            batchResultSection.innerHTML = `<div class="empty-state"><i class="fas fa-exclamation-triangle"></i><p>CSV upload failed.</p></div>`;
        });
    });

    function renderBatchResults(predictions) {
        if (!predictions || predictions.length === 0) {
            batchResultSection.innerHTML = `<div class="empty-state"><p>No rows processed.</p></div>`;
            return;
        }
        
        let html = `
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                    <h3 class="chart-title">Batch Results (${predictions.length} Rows Processed)</h3>
                    <button class="btn btn-secondary" id="downloadBtn" style="padding: 0.5rem 1rem; font-size: 0.85rem;">
                        <i class="fas fa-download"></i> Download Annotated CSV
                    </button>
                </div>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th style="width: 60%">Original Tweet</th>
                                <th style="width: 20%">Sentiment</th>
                                <th style="width: 20%">Confidence</th>
                            </tr>
                        </thead>
                        <tbody>
        `;
        
        predictions.forEach(row => {
            const badgeClass = row.sentiment === "Positive" ? "badge-positive" : (row.sentiment === "Neutral" ? "badge-neutral" : "badge-negative");
            html += `
                <tr>
                    <td style="word-break: break-all;">${escapeHtml(row.original_tweet)}</td>
                    <td><span class="badge ${badgeClass}">${row.sentiment}</span></td>
                    <td>${row.confidence.toFixed(1)}%</td>
                </tr>
            `;
        });
        
        html += `
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        
        batchResultSection.innerHTML = html;
        
        // Re-attach download listener
        document.getElementById("downloadBtn").addEventListener("click", () => {
            if (lastBatchFilename) {
                window.location.href = `/uploads/${lastBatchFilename}`;
            }
        });
    }

    // 3. Load Overview Statistics
    function loadOverviewStats() {
        fetch("/dashboard-stats")
        .then(res => res.json())
        .then(data => {
            document.getElementById("totalAnalyzedVal").textContent = data.db_stats.total;
            document.getElementById("posVal").textContent = data.db_stats.distribution.Positive;
            document.getElementById("neuVal").textContent = data.db_stats.distribution.Neutral;
            document.getElementById("negVal").textContent = data.db_stats.distribution.Negative;
            
            // Set active model badge on load
            if (data.model_metadata && data.model_metadata.best_model) {
                document.getElementById("modelNameBadge").textContent = `Model: ${data.model_metadata.best_model}`;
            }
        })
        .catch(err => console.error("Error loading overview stats:", err));
    }

    // 4. Load Dashboard Visualization Data
    function loadDashboardData() {
        fetch("/dashboard-stats")
        .then(res => res.json())
        .then(data => {
            modelMetadata = data.model_metadata;
            
            // Set active model badge
            if (modelMetadata && modelMetadata.best_model) {
                document.getElementById("modelNameBadge").textContent = `Model: ${modelMetadata.best_model}`;
            }
            
            // A. Render Charts
            renderSentimentDistributionChart(data.db_stats.distribution);
            
            if (modelMetadata) {
                renderModelComparisonChart(modelMetadata.models_comparison);
                renderConfusionMatrix(modelMetadata.confusion_matrix);
                renderCommonWords(modelMetadata.top_words);
            }
        })
        .catch(err => console.error("Error loading dashboard data:", err));
    }

    function renderSentimentDistributionChart(distribution) {
        const ctx = document.getElementById("sentimentDonutChart").getContext("2d");
        
        const dataValues = [distribution.Positive, distribution.Neutral, distribution.Negative];
        
        if (donutChartInstance) {
            donutChartInstance.destroy();
        }
        
        donutChartInstance = new Chart(ctx, {
            type: "doughnut",
            data: {
                labels: ["Positive", "Neutral", "Negative"],
                datasets: [{
                    data: dataValues,
                    backgroundColor: ["#10b981", "#64748b", "#f43f5e"],
                    borderWidth: 1,
                    borderColor: "rgba(255, 255, 255, 0.08)"
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "bottom",
                        labels: {
                            color: "#9ca3af",
                            font: { family: "Inter" }
                        }
                    }
                },
                cutout: "65%"
            }
        });
    }

    function renderModelComparisonChart(comparison) {
        if (!comparison) return;
        const ctx = document.getElementById("modelComparisonChart").getContext("2d");
        
        const labels = Object.keys(comparison);
        const accuracies = labels.map(l => (comparison[l].accuracy * 100).toFixed(1));
        const f1Scores = labels.map(l => (comparison[l].f1_score * 100).toFixed(1));
        
        if (comparisonChartInstance) {
            comparisonChartInstance.destroy();
        }
        
        comparisonChartInstance = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "Accuracy (%)",
                        data: accuracies,
                        backgroundColor: "rgba(6, 182, 212, 0.65)",
                        borderColor: "rgba(6, 182, 212, 1)",
                        borderWidth: 1,
                        borderRadius: 4
                    },
                    {
                        label: "F1-Score (%)",
                        data: f1Scores,
                        backgroundColor: "rgba(99, 102, 241, 0.65)",
                        borderColor: "rgba(99, 102, 241, 1)",
                        borderWidth: 1,
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#9ca3af" }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: "#9ca3af" }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: "#9ca3af", font: { family: "Inter" } }
                    }
                }
            }
        });
    }

    function renderConfusionMatrix(matrixData) {
        if (!matrixData) return;
        
        const matrix = matrixData.matrix; // Array of arrays [[NegNeg, NegNeu, NegPos], [NeuNeg, ...]]
        const cells = [
            document.getElementById("cm-0-0"), document.getElementById("cm-0-1"), document.getElementById("cm-0-2"),
            document.getElementById("cm-1-0"), document.getElementById("cm-1-1"), document.getElementById("cm-1-2"),
            document.getElementById("cm-2-0"), document.getElementById("cm-2-1"), document.getElementById("cm-2-2")
        ];
        
        // Flatten matrix values
        const flatValues = matrix.flat();
        const maxVal = Math.max(...flatValues) || 1;
        
        let index = 0;
        for (let row = 0; row < 3; row++) {
            for (let col = 0; col < 3; col++) {
                const val = matrix[row][col];
                const cell = cells[index];
                cell.textContent = val;
                
                // Remove existing heat-color classes
                cell.className = "matrix-cell matrix-data-cell";
                
                // Color scale based on proportion of maximum cell count
                const ratio = val / maxVal;
                if (ratio > 0.6) {
                    cell.classList.add("cell-high");
                } else if (ratio > 0.2) {
                    cell.classList.add("cell-med");
                } else if (val > 0) {
                    cell.classList.add("cell-low");
                }
                
                index++;
            }
        }
    }

    function renderCommonWords(topWords) {
        if (!topWords) return;
        
        const posList = document.getElementById("posWordsList");
        const negList = document.getElementById("negWordsList");
        
        // Positives
        posList.innerHTML = "";
        const maxPos = topWords.Positive.length > 0 ? topWords.Positive[0].count : 1;
        topWords.Positive.slice(0, 7).forEach(item => {
            const pct = (item.count / maxPos) * 100;
            posList.innerHTML += `
                <div class="word-row">
                    <span class="word-name">#${item.word}</span>
                    <div class="word-bar-container">
                        <div class="word-bar-outer">
                            <div class="word-bar-inner pos" style="width: ${pct}%"></div>
                        </div>
                        <span class="word-count-label">${item.count}</span>
                    </div>
                </div>
            `;
        });
        
        // Negatives
        negList.innerHTML = "";
        const maxNeg = topWords.Negative.length > 0 ? topWords.Negative[0].count : 1;
        topWords.Negative.slice(0, 7).forEach(item => {
            const pct = (item.count / maxNeg) * 100;
            negList.innerHTML += `
                <div class="word-row">
                    <span class="word-name">#${item.word}</span>
                    <div class="word-bar-container">
                        <div class="word-bar-outer">
                            <div class="word-bar-inner neg" style="width: ${pct}%"></div>
                        </div>
                        <span class="word-count-label">${item.count}</span>
                    </div>
                </div>
            `;
        });
    }

    // 5. Load Prediction History
    function loadHistory() {
        const sentimentVal = filterSentiment.value;
        const searchVal = searchHistory.value.trim();
        
        let url = "/dashboard-stats";
        // Build queries
        const queryParams = [];
        if (sentimentVal) queryParams.push(`sentiment=${encodeURIComponent(sentimentVal)}`);
        if (searchVal) queryParams.push(`search=${encodeURIComponent(searchVal)}`);
        
        if (queryParams.length) {
            url += `?${queryParams.join("&")}`;
        }
        
        fetch(url)
        .then(res => res.json())
        .then(data => {
            renderHistoryRows(data.history);
        })
        .catch(err => console.error("Error loading history:", err));
    }

    function renderHistoryRows(history) {
        if (!history || history.length === 0) {
            historyTableBody.innerHTML = `
                <tr>
                    <td colspan="5" class="empty-state">
                        <i class="far fa-folder-open"></i>
                        <p>No history records found.</p>
                    </td>
                </tr>
            `;
            return;
        }
        
        let html = "";
        history.forEach(row => {
            const badgeClass = row.sentiment === "Positive" ? "badge-positive" : (row.sentiment === "Neutral" ? "badge-neutral" : "badge-negative");
            
            // Format Timestamp
            let dateStr = row.timestamp;
            try {
                const dt = new Date(row.timestamp);
                dateStr = dt.toLocaleString();
            } catch(e) {}
            
            html += `
                <tr>
                    <td>${dateStr}</td>
                    <td style="max-width: 350px; word-wrap: break-word;">${escapeHtml(row.original_text)}</td>
                    <td><span class="badge ${badgeClass}">${row.sentiment}</span></td>
                    <td>${row.confidence.toFixed(1)}%</td>
                    <td>
                        <button class="btn btn-danger delete-btn" data-id="${row.id}" style="padding: 0.35rem 0.75rem; font-size: 0.75rem; border-radius: 6px;">
                            <i class="far fa-trash-alt"></i> Delete
                        </button>
                    </td>
                </tr>
            `;
        });
        
        historyTableBody.innerHTML = html;
        
        // Add Delete Listeners
        const deleteButtons = historyTableBody.querySelectorAll(".delete-btn");
        deleteButtons.forEach(btn => {
            btn.addEventListener("click", () => {
                const id = btn.dataset.id;
                deleteHistoryEntry(id);
            });
        });
    }

    function deleteHistoryEntry(id) {
        if (!confirm("Are you sure you want to delete this prediction from history?")) return;
        
        fetch(`/delete/${id}`, {
            method: "POST"
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                loadHistory();
                loadOverviewStats();
            } else {
                alert("Failed to delete record.");
            }
        })
        .catch(err => console.error("Error deleting history row:", err));
    }

    // Filter listeners
    filterSentiment.addEventListener("change", loadHistory);
    
    // Search history input listener with simple debounce
    let searchTimeout = null;
    searchHistory.addEventListener("input", () => {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
            loadHistory();
        }, 300);
    });

    // Clear All History
    clearHistoryBtn.addEventListener("click", () => {
        if (!confirm("Are you sure you want to CLEAR ALL history from the database? This cannot be undone.")) return;
        
        fetch("/clear-history", {
            method: "POST"
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                loadHistory();
                loadOverviewStats();
            } else {
                alert("Failed to clear database history.");
            }
        })
        .catch(err => console.error("Error clearing history database:", err));
    });

    // Helper: Escape HTML to avoid XSS issues
    function escapeHtml(text) {
        if (!text) return "";
        return text
            .toString()
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
