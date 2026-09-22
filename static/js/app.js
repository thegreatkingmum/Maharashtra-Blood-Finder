document.addEventListener("DOMContentLoaded", () => {
    initializePage();
});

async function initializePage() {
    const path = window.location.pathname;

    try {
        if (path === "/risk") {
            await initializeRiskPage();
        } else if (path === "/insights") {
            await initializeInsightsPage();
        } else if (path === "/finder" || path === "/find-blood") {
            await initializeFinderPage();
        } else {
            await initializeGeneralPage();
        }
    } catch (error) {
        console.error("Page initialization error:", error);
    }
}

/* =========================================================
   COMMON HELPERS
========================================================= */

async function fetchJSON(url, options = {}) {
    const response = await fetch(url, {
        headers: {
            "Accept": "application/json"
        },
        ...options
    });

    if (!response.ok) {
        throw new Error(`${url} returned HTTP ${response.status}`);
    }

    return await response.json();
}

function firstExistingElement(ids) {
    for (const id of ids) {
        const element = document.getElementById(id);
        if (element) return element;
    }
    return null;
}

function setText(ids, value) {
    const element = firstExistingElement(ids);
    if (element) {
        element.textContent = value ?? "—";
    }
}

function numberValue(value, decimals = 2) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "—";
    }

    return number.toFixed(decimals);
}

function percentValue(value, decimals = 1) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "—";
    }

    return `${number.toFixed(decimals)}%`;
}

function escapeHTML(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function findArray(data, possibleKeys = []) {
    if (Array.isArray(data)) {
        return data;
    }

    if (!data || typeof data !== "object") {
        return [];
    }

    for (const key of possibleKeys) {
        if (Array.isArray(data[key])) {
            return data[key];
        }
    }

    for (const value of Object.values(data)) {
        if (Array.isArray(value)) {
            return value;
        }
    }

    return [];
}

function findValue(object, possibleKeys = [], defaultValue = null) {
    if (!object || typeof object !== "object") {
        return defaultValue;
    }

    for (const key of possibleKeys) {
        if (
            Object.prototype.hasOwnProperty.call(object, key) &&
            object[key] !== null &&
            object[key] !== undefined
        ) {
            return object[key];
        }
    }

    return defaultValue;
}

/* =========================================================
   RISK PAGE
========================================================= */

async function initializeRiskPage() {
    console.log("Initializing Risk & Prediction page...");

    const tasks = [
        loadBBRISummary(),
        loadBBRITable(),
        loadModelComparison(),
        loadRiskBanks(),
        initializePrediction()
    ];

    const results = await Promise.allSettled(tasks);

    results.forEach((result, index) => {
        if (result.status === "rejected") {
            console.error(`Risk section ${index + 1} failed:`, result.reason);
        }
    });

    console.log("Risk page initialization complete.");
}

/* =========================================================
   BBRI SUMMARY
========================================================= */

async function loadBBRISummary() {
    try {
        const data = await fetchJSON("/api/risk/bbri-summary");

        console.log("BBRI summary:", data);

        const rows = findArray(data, [
            "data",
            "rows",
            "bbri",
            "banks"
        ]);

        let count = findValue(data, [
            "blood_banks",
            "bank_count",
            "count",
            "total_banks"
        ]);

        let average = findValue(data, [
            "average_bbri",
            "avg_bbri",
            "mean_bbri"
        ]);

        let median = findValue(data, [
            "median_bbri"
        ]);

        let highest = findValue(data, [
            "highest_bbri",
            "max_bbri"
        ]);

        if (rows.length > 0) {
            const values = rows
                .map(row =>
                    Number(
                        findValue(row, [
                            "BBRI",
                            "bbri",
                            "bbri_score"
                        ])
                    )
                )
                .filter(Number.isFinite);

            if (count === null) count = rows.length;

            if (values.length > 0) {
                if (average === null) {
                    average =
                        values.reduce((sum, value) => sum + value, 0) /
                        values.length;
                }

                if (median === null) {
                    const sorted = [...values].sort((a, b) => a - b);
                    const middle = Math.floor(sorted.length / 2);

                    median =
                        sorted.length % 2 === 0
                            ? (sorted[middle - 1] + sorted[middle]) / 2
                            : sorted[middle];
                }

                if (highest === null) {
                    highest = Math.max(...values);
                }
            }
        }

        setText(
            ["bbriBankCount", "riskBankCount", "totalRiskBanks"],
            count !== null ? numberValue(count, 0) : "—"
        );

        setText(
            ["avgBBRI", "averageBBRI", "bbriAverage"],
            average !== null ? numberValue(average, 2) : "—"
        );

        setText(
            ["medianBBRI", "bbriMedian"],
            median !== null ? numberValue(median, 2) : "—"
        );

        setText(
            ["highestBBRI", "bbriHighest", "maxBBRI"],
            highest !== null ? numberValue(highest, 2) : "—"
        );
    } catch (error) {
        console.error("BBRI summary error:", error);
    }
}

/* =========================================================
   BBRI TABLE
========================================================= */

async function loadBBRITable() {
    try {
        const data = await fetchJSON("/api/risk/bbri");

        console.log("BBRI data:", data);

        const rows = findArray(data, [
            "data",
            "rows",
            "bbri",
            "banks"
        ]);

        const tableBody = firstExistingElement([
            "bbriTableBody",
            "bbri-table-body",
            "riskTableBody"
        ]);

        if (!tableBody) {
            console.warn("BBRI table body not found.");
            return;
        }

        if (rows.length === 0) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="20" class="text-center text-muted">
                        No BBRI data available.
                    </td>
                </tr>
            `;
            return;
        }

        tableBody.innerHTML = rows.map((row, index) => {
            const code = findValue(row, [
                "hospital_code",
                "Hospital Code",
                "code"
            ], "—");

            const name = findValue(row, [
                "blood_bank_name",
                "Blood Bank Name",
                "name"
            ], "—");

            const district = findValue(row, [
                "district",
                "District"
            ], "—");

            const city = findValue(row, [
                "city",
                "City"
            ], "—");

            const area = findValue(row, [
                "area",
                "Area"
            ], "—");

            const bbri = findValue(row, [
                "BBRI",
                "bbri",
                "bbri_score"
            ]);

            const riskClass = findValue(row, [
                "risk_class",
                "risk_category",
                "Risk Category"
            ], "—");

            const stockout = findValue(row, [
                "stockout_frequency",
                "stockout_percent",
                "stockout_percentage",
                "Stockout %"
            ]);

            const lowStock = findValue(row, [
                "low_stock_frequency",
                "low_stock_percent",
                "low_stock_percentage",
                "Low-stock %"
            ]);

            const volatility = findValue(row, [
                "stock_volatility",
                "volatility"
            ]);

            const recovery = findValue(row, [
                "recovery_rate",
                "Recovery Rate"
            ]);

            let badgeClass = "bg-secondary";

            const normalizedRisk = String(riskClass).toLowerCase();

            if (
                normalizedRisk.includes("very high") ||
                normalizedRisk.includes("critical")
            ) {
                badgeClass = "bg-danger";
            } else if (normalizedRisk.includes("high")) {
                badgeClass = "bg-warning text-dark";
            } else if (normalizedRisk.includes("moderate")) {
                badgeClass = "bg-info text-dark";
            } else if (normalizedRisk.includes("low")) {
                badgeClass = "bg-success";
            }

            return `
                <tr>
                    <td>${index + 1}</td>
                    <td>${escapeHTML(code)}</td>
                    <td>${escapeHTML(name)}</td>
                    <td>${escapeHTML(district)}</td>
                    <td>${escapeHTML(city)}</td>
                    <td>${escapeHTML(area)}</td>
                    <td><strong>${numberValue(bbri, 2)}</strong></td>
                    <td>
                        <span class="badge ${badgeClass}">
                            ${escapeHTML(riskClass)}
                        </span>
                    </td>
                    <td>${percentValue(stockout)}</td>
                    <td>${percentValue(lowStock)}</td>
                    <td>${numberValue(volatility, 3)}</td>
                    <td>${percentValue(recovery)}</td>
                </tr>
            `;
        }).join("");
    } catch (error) {
        console.error("BBRI table error:", error);
    }
}

/* =========================================================
   RISK BANK DROPDOWN
========================================================= */

async function loadRiskBanks() {
    try {
        const data = await fetchJSON("/api/risk/banks");

        console.log("Risk banks:", data);

        const rows = findArray(data, [
            "data",
            "rows",
            "banks"
        ]);

        const select = firstExistingElement([
            "riskBankSelect",
            "bankSelect",
            "predictionBank"
        ]);

        if (!select) {
            console.warn("Risk bank dropdown not found.");
            return;
        }

        select.innerHTML = `
            <option value="">Select a blood bank</option>
        `;

        rows.forEach(row => {
            const code = findValue(row, [
                "hospital_code",
                "Hospital Code",
                "code"
            ]);

            const name = findValue(row, [
                "blood_bank_name",
                "Blood Bank Name",
                "name"
            ], "Blood Bank");

            if (code !== null && code !== undefined) {
                const option = document.createElement("option");
                option.value = code;
                option.textContent = `${name} (${code})`;
                select.appendChild(option);
            }
        });
    } catch (error) {
        console.error("Risk banks error:", error);
    }
}

/* =========================================================
   MODEL COMPARISON
========================================================= */

async function loadModelComparison() {
    try {
        const data = await fetchJSON("/api/risk/model-comparison");

        console.log("Model comparison:", data);

        const rows = findArray(data, [
            "data",
            "rows",
            "models"
        ]);

        if (rows.length === 0) {
            return;
        }

        const tableBody = firstExistingElement([
            "modelComparisonBody",
            "modelTableBody",
            "comparisonTableBody"
        ]);

        if (!tableBody) {
            return;
        }

        tableBody.innerHTML = rows.map(row => {
            const model = findValue(row, [
                "model",
                "Model",
                "name"
            ], "—");

            const accuracy = findValue(row, [
                "accuracy",
                "Accuracy"
            ]);

            const precision = findValue(row, [
                "precision",
                "Precision"
            ]);

            const recall = findValue(row, [
                "recall",
                "Recall"
            ]);

            const f1 = findValue(row, [
                "f1",
                "f1_score",
                "F1"
            ]);

            const rocAuc = findValue(row, [
                "roc_auc",
                "roc_auc_score",
                "ROC-AUC"
            ]);

            return `
                <tr>
                    <td>${escapeHTML(model)}</td>
                    <td>${percentValue(Number(accuracy) * 100)}</td>
                    <td>${percentValue(Number(precision) * 100)}</td>
                    <td>${percentValue(Number(recall) * 100)}</td>
                    <td>${percentValue(Number(f1) * 100)}</td>
                    <td>${percentValue(Number(rocAuc) * 100)}</td>
                </tr>
            `;
        }).join("");
    } catch (error) {
        console.error("Model comparison error:", error);
    }
}

/* =========================================================
   FEATURE IMPORTANCE
========================================================= */

async function loadFeatureImportance() {
    try {
        const data = await fetchJSON("/api/risk/feature-importance");

        console.log("Feature importance:", data);

        const rows = findArray(data, [
            "data",
            "rows",
            "features",
            "importance"
        ]);

        const tableBody = firstExistingElement([
            "featureImportanceBody",
            "featureTableBody",
            "importanceTableBody"
        ]);

        if (tableBody && rows.length > 0) {
            tableBody.innerHTML = rows.map((row, index) => {
                const feature = findValue(row, [
                    "feature",
                    "Feature",
                    "name"
                ], "—");

                const importance = findValue(row, [
                    "importance",
                    "permutation_importance",
                    "mean_importance",
                    "Mean Importance"
                ]);

                return `
                    <tr>
                        <td>${index + 1}</td>
                        <td>${escapeHTML(feature)}</td>
                        <td>${numberValue(importance, 4)}</td>
                    </tr>
                `;
            }).join("");
        }

        const canvas = firstExistingElement([
            "featureImportanceChart",
            "importanceChart"
        ]);

        if (
            canvas &&
            rows.length > 0 &&
            typeof Chart !== "undefined"
        ) {
            const labels = rows.map(row =>
                findValue(row, [
                    "feature",
                    "Feature",
                    "name"
                ], "Unknown")
            );

            const values = rows.map(row =>
                Number(findValue(row, [
                    "importance",
                    "permutation_importance",
                    "mean_importance"
                ], 0))
            );

            new Chart(canvas, {
                type: "bar",
                data: {
                    labels,
                    datasets: [{
                        label: "Feature Importance",
                        data: values
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        x: {
                            ticks: {
                                autoSkip: false
                            }
                        }
                    }
                }
            });
        }
    } catch (error) {
        console.error("Feature importance error:", error);
    }
}

/* =========================================================
   PREDICTION
========================================================= */

async function initializePrediction() {
    const predictButton = firstExistingElement([
        "predictButton",
        "runPrediction",
        "predictionButton"
    ]);

    if (!predictButton) {
        console.warn("Prediction button not found.");
        return;
    }

    predictButton.addEventListener("click", async () => {
        await runPrediction();
    });
}

async function runPrediction() {
    const bankSelect = firstExistingElement([
        "riskBankSelect",
        "bankSelect",
        "predictionBank"
    ]);

    const groupSelect = firstExistingElement([
        "riskBloodGroupSelect",
        "bloodGroupSelect",
        "predictionBloodGroup"
    ]);

    if (!bankSelect || !groupSelect) {
        showPredictionMessage(
            "Prediction controls could not be found."
        );
        return;
    }

    const hospitalCode = bankSelect.value;
    const bloodGroup = groupSelect.value;

    if (!hospitalCode || !bloodGroup) {
        showPredictionMessage(
            "Please select a blood bank and blood group."
        );
        return;
    }

    showPredictionMessage("Calculating prediction...");

    try {
        const data = await fetchJSON("/api/risk/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            body: JSON.stringify({
                hospital_code: hospitalCode,
                blood_group: bloodGroup
            })
        });

        console.log("Prediction:", data);

        renderPredictionResult(data);
    } catch (error) {
        console.error("Prediction error:", error);
        showPredictionMessage(
            "Prediction failed. Check that the HistGradientBoosting model is available."
        );
    }
}

function renderPredictionResult(data) {
    const probability = findValue(data, [
        "probability",
        "stockout_probability",
        "predicted_probability",
        "risk_probability"
    ]);

    const risk = findValue(data, [
        "risk",
        "risk_label",
        "risk_class",
        "prediction"
    ], "Unknown");

    const bankName = findValue(data, [
        "blood_bank_name",
        "bank_name",
        "hospital_name"
    ], "");

    const bloodGroup = findValue(data, [
        "blood_group",
        "group"
    ], "");

    const reasons = findArray(data, [
        "reasons",
        "explanations",
        "drivers"
    ]);

    const resultContainer = firstExistingElement([
        "predictionResult",
        "prediction-result",
        "riskPredictionResult"
    ]);

    if (!resultContainer) {
        console.warn("Prediction result container not found.");
        return;
    }

    const numericProbability = Number(probability);

    let displayProbability = "—";

    if (Number.isFinite(numericProbability)) {
        displayProbability =
            numericProbability <= 1
                ? `${(numericProbability * 100).toFixed(1)}%`
                : `${numericProbability.toFixed(1)}%`;
    }

    const normalizedRisk = String(risk).toLowerCase();

    let badgeClass = "bg-secondary";

    if (
        normalizedRisk.includes("high") ||
        normalizedRisk.includes("critical")
    ) {
        badgeClass = "bg-danger";
    } else if (normalizedRisk.includes("moderate")) {
        badgeClass = "bg-warning text-dark";
    } else if (normalizedRisk.includes("low")) {
        badgeClass = "bg-success";
    }

    let reasonsHTML = "";

    resultContainer.innerHTML = `
        <div class="card border-0 shadow-sm">
            <div class="card-body">
                <h5 class="mb-3">
                    Stockout Prediction
                </h5>

                ${bankName ? `
                    <div>
                        <strong>Blood Bank:</strong>
                        ${escapeHTML(bankName)}
                    </div>
                ` : ""}

                ${bloodGroup ? `
                    <div>
                        <strong>Blood Group:</strong>
                        ${escapeHTML(bloodGroup)}
                    </div>
                ` : ""}

                <div class="mt-3">
                    <strong>Predicted Stockout Probability:</strong>
                    <span class="fs-4">
                        ${displayProbability}
                    </span>
                </div>

                <div class="mt-2">
                    <strong>Risk Level:</strong>
                    <span class="badge ${badgeClass}">
                        ${escapeHTML(risk)}
                    </span>
                </div>

                ${reasonsHTML}
            </div>
        </div>
    `;
}

function showPredictionMessage(message) {
    const resultContainer = firstExistingElement([
        "predictionResult",
        "prediction-result",
        "riskPredictionResult"
    ]);

    if (resultContainer) {
        resultContainer.innerHTML = `
            <div class="alert alert-info">
                ${escapeHTML(message)}
            </div>
        `;
    }
}

/* =========================================================
   INSIGHTS PAGE
========================================================= */

async function initializeInsightsPage() {
    console.log("Initializing Insights page...");

    const endpoints = [
        "/api/insights/overview",
        "/api/insights/availability",
        "/api/insights/district-stockout",
        "/api/insights/hotspots",
        "/api/insights/blood-group-comparison"
    ];

    for (const endpoint of endpoints) {
        try {
            const data = await fetchJSON(endpoint);
            console.log(endpoint, data);
        } catch (error) {
            console.error(endpoint, error);
        }
    }

    console.log("Insights page initialization complete.");
}

/* =========================================================
   FINDER PAGE
========================================================= */

async function initializeFinderPage() {
    console.log("Initializing Finder page...");

    try {
        await loadLocations();
    } catch (error) {
        console.error("Finder initialization error:", error);
    }

    const searchButton = firstExistingElement([
        "findBloodButton",
        "searchBloodButton",
        "searchButton"
    ]);

    if (searchButton) {
        searchButton.addEventListener("click", async () => {
            await searchBlood();
        });
    }
}

async function loadLocations() {
    const data = await fetchJSON("/api/locations");

    console.log("Locations:", data);

    const districts = findArray(data, [
        "districts"
    ]);

    const cities = findArray(data, [
        "cities"
    ]);

    const districtSelect = firstExistingElement([
        "districtSelect"
    ]);

    const citySelect = firstExistingElement([
        "citySelect"
    ]);

    if (districtSelect && districts.length > 0) {
        districtSelect.innerHTML = `
            <option value="">All districts</option>
        `;

        districts.forEach(item => {
            const value =
                typeof item === "string"
                    ? item
                    : findValue(item, [
                        "district",
                        "name",
                        "value"
                    ], "");

            if (value) {
                const option = document.createElement("option");
                option.value = value;
                option.textContent = value;
                districtSelect.appendChild(option);
            }
        });
    }

    if (citySelect && cities.length > 0) {
        citySelect.innerHTML = `
            <option value="">All cities</option>
        `;

        cities.forEach(item => {
            const value =
                typeof item === "string"
                    ? item
                    : findValue(item, [
                        "city",
                        "name",
                        "value"
                    ], "");

            if (value) {
                const option = document.createElement("option");
                option.value = value;
                option.textContent = value;
                citySelect.appendChild(option);
            }
        });
    }
}

async function searchBlood() {
    const areaInput = firstExistingElement([
        "areaInput",
        "area"
    ]);

    const citySelect = firstExistingElement([
        "citySelect"
    ]);

    const districtSelect = firstExistingElement([
        "districtSelect"
    ]);

    const bloodGroupSelect = firstExistingElement([
        "bloodGroupSelect",
        "bloodGroup"
    ]);

    const unitsInput = firstExistingElement([
        "requiredUnits",
        "unitsInput"
    ]);

    const params = new URLSearchParams();

    if (areaInput?.value.trim()) {
        params.set("area", areaInput.value.trim());
    }

    if (citySelect?.value) {
        params.set("city", citySelect.value);
    }

    if (districtSelect?.value) {
        params.set("district", districtSelect.value);
    }

    if (bloodGroupSelect?.value) {
        params.set("blood_group", bloodGroupSelect.value);
    }

    if (unitsInput?.value) {
        params.set("required_units", unitsInput.value);
    }

    try {
        const data = await fetchJSON(
            `/api/find-blood?${params.toString()}`
        );

        renderFinderResults(data);
    } catch (error) {
        console.error("Blood finder error:", error);
    }
}

function renderFinderResults(data) {
    const container = firstExistingElement([
        "resultsContainer",
        "bloodResults",
        "finderResults"
    ]);

    if (!container) {
        return;
    }

    const rows = findArray(data, [
        "results",
        "data",
        "banks"
    ]);

    if (rows.length === 0) {
        container.innerHTML = `
            <div class="alert alert-warning">
                No matching blood banks found.
            </div>
        `;
        return;
    }

    container.innerHTML = rows.map(row => {
        const name = findValue(row, [
            "blood_bank_name",
            "name"
        ], "Blood Bank");

        const area = findValue(row, [
            "area"
        ], "");

        const city = findValue(row, [
            "city"
        ], "");

        const district = findValue(row, [
            "district"
        ], "");

        const stock = findValue(row, [
            "stock"
        ], "");

        const phone = findValue(row, [
            "phone",
            "hospitalcontact",
            "contact"
        ], "");

        const latitude = findValue(row, [
            "latitude",
            "lat"
        ]);

        const longitude = findValue(row, [
            "longitude",
            "lon",
            "lng"
        ]);

        let mapsURL = "";

        if (
            Number.isFinite(Number(latitude)) &&
            Number.isFinite(Number(longitude)) &&
            Number(latitude) !== 0 &&
            Number(longitude) !== 0
        ) {
            mapsURL =
                `https://www.google.com/maps?q=${latitude},${longitude}`;
        }

        return `
            <div class="card mb-3 shadow-sm">
                <div class="card-body">
                    <h5>${escapeHTML(name)}</h5>

                    <div class="text-muted">
                        ${escapeHTML(area)}
                        ${area && city ? ", " : ""}
                        ${escapeHTML(city)}
                        ${city && district ? ", " : ""}
                        ${escapeHTML(district)}
                    </div>

                    <div class="mt-2">
                        <strong>Stock:</strong>
                        ${escapeHTML(stock)}
                    </div>

                    <div class="mt-3 d-flex gap-2">
                        ${phone ? `
                            <a
                                class="btn btn-outline-danger"
                                href="tel:${escapeHTML(phone)}"
                            >
                                Call
                            </a>
                        ` : ""}

                        ${mapsURL ? `
                            <a
                                class="btn btn-outline-dark"
                                href="${mapsURL}"
                                target="_blank"
                                rel="noopener"
                            >
                                View on Google Maps
                            </a>
                        ` : ""}
                    </div>
                </div>
            </div>
        `;
    }).join("");
}

/* =========================================================
   GENERAL
========================================================= */

async function initializeGeneralPage() {
    // Intentionally empty.
    // This keeps base.html from producing errors on pages
    // that don't need page-specific JavaScript.
}