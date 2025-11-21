window.addEventListener("load", () => {
  loadAccuracy();
  loadConfusionMatrix();
  loadROC();
  loadPR();
  loadFeatureImportance();
  setupPredict();
});

const API_BASE = "http://127.0.0.1:8000";

// Fetch and show accuracy
async function loadAccuracy() {
  try {
    const res = await fetch(`${API_BASE}/metrics`);
    const data = await res.json();
    document.getElementById("modelAccuracy").innerText = data.test_accuracy
      ? `Model Test Accuracy: ${data.test_accuracy}%`
      : "Model Accuracy: N/A";
  } catch (e) {
    console.error("Accuracy fetch error", e);
  }
}

// ----------------- Confusion Matrix -----------------
async function loadConfusionMatrix() {
  try {
    const res = await fetch(`${API_BASE}/confusion_matrix`);
    const json = await res.json();
    const matrix = json.confusion_matrix || [
      [0, 0],
      [0, 0],
    ];
    const maxVal = Math.max(...matrix.flat(), 1);

    const values = [
      { x: 0, y: 0, v: matrix[0][0] },
      { x: 1, y: 0, v: matrix[0][1] },
      { x: 0, y: 1, v: matrix[1][0] },
      { x: 1, y: 1, v: matrix[1][1] },
    ];

    const ctx = document.getElementById("confMatrixChart").getContext("2d");
    if (ctx._chart) ctx._chart.destroy();

    ctx._chart = new Chart(ctx, {
      type: "matrix",
      data: {
        datasets: [
          {
            label: "Confusion Matrix",
            data: values,
            backgroundColor: (ctxArg) => {
              const intensity =
                ctxArg.dataset.data[ctxArg.dataIndex].v / maxVal;
              return `rgba(0,120,255,${0.2 + 0.8 * intensity})`;
            },
            borderColor: "rgba(0,0,0,0.6)",
            borderWidth: 1,
            width: 140,
            height: 140,
          },
        ],
      },
      options: {
        responsive: false,
        plugins: {
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const mapping = [
                  "Actual Legitimate → Predicted Legitimate",
                  "Actual Legitimate → Predicted Phishing",
                  "Actual Phishing → Predicted Legitimate",
                  "Actual Phishing → Predicted Phishing",
                ];
                return `${mapping[ctx.dataIndex]}: ${ctx.raw.v}`;
              },
            },
          },
          legend: { display: false },
        },
        scales: {
          x: {
            type: "linear",
            min: -0.5,
            max: 1.5,
            ticks: {
              stepSize: 1,
              callback: (v) => ["Pred Legitimate", "Pred Phishing"][v] || v,
            },
            grid: { display: false },
          },
          y: {
            type: "linear",
            min: -0.5,
            max: 1.5,
            ticks: {
              stepSize: 1,
              callback: (v) => ["Actual Legitimate", "Actual Phishing"][v] || v,
            },
            grid: { display: false },
          },
        },
      },
    });
  } catch (e) {
    console.error("Confusion matrix load error", e);
  }
}

// ----------------- ROC Curve -----------------
async function loadROC() {
  try {
    const res = await fetch(`${API_BASE}/roc_curve`);
    const json = await res.json();
    const fpr = json.fpr || [];
    const tpr = json.tpr || [];

    const ctx = document.getElementById("rocChart").getContext("2d");
    if (ctx._chart) ctx._chart.destroy();

    ctx._chart = new Chart(ctx, {
      type: "line",
      data: {
        datasets: [
          {
            label: "ROC (TPR vs FPR)",
            data: fpr.map((x, i) => ({
              x: isFinite(x) ? x : 0,
              y: isFinite(tpr[i]) ? tpr[i] : 0,
            })),
            showLine: true,
            fill: false,
            tension: 0.1,
          },
        ],
      },
      options: {
        parsing: { xAxisKey: "x", yAxisKey: "y" },
        scales: {
          x: {
            title: { display: true, text: "False Positive Rate" },
            min: 0,
            max: 1,
          },
          y: {
            title: { display: true, text: "True Positive Rate" },
            min: 0,
            max: 1,
          },
        },
      },
    });
  } catch (e) {
    console.error("ROC load error", e);
  }
}

// ----------------- Precision-Recall -----------------
async function loadPR() {
  try {
    const res = await fetch(`${API_BASE}/precision_recall`);
    const json = await res.json();
    const precision = json.precision || [];
    const recall = json.recall || [];

    const ctx = document.getElementById("prChart").getContext("2d");
    if (ctx._chart) ctx._chart.destroy();

    ctx._chart = new Chart(ctx, {
      type: "line",
      data: {
        datasets: [
          {
            label: "Precision vs Recall",
            data: recall.map((r, i) => ({
              x: isFinite(r) ? r : 0,
              y: isFinite(precision[i]) ? precision[i] : 0,
            })),
            showLine: true,
            fill: false,
            tension: 0.1,
          },
        ],
      },
      options: {
        parsing: { xAxisKey: "x", yAxisKey: "y" },
        scales: {
          x: { title: { display: true, text: "Recall" }, min: 0, max: 1 },
          y: { title: { display: true, text: "Precision" }, min: 0, max: 1 },
        },
      },
    });
  } catch (e) {
    console.error("PR load error", e);
  }
}

// ----------------- Feature Importance -----------------
async function loadFeatureImportance() {
  try {
    const res = await fetch(`${API_BASE}/feature_importance`);
    const json = await res.json();
    const top = json.top_features || [];
    const top20 = top.slice().reverse();

    const ctx = document.getElementById("fiChart").getContext("2d");
    if (ctx._chart) ctx._chart.destroy();

    ctx._chart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: top20.map((f) => f.feature),
        datasets: [
          { label: "Importance", data: top20.map((f) => f.importance) },
        ],
      },
      options: {
        indexAxis: "y",
        plugins: { legend: { display: false } },
        scales: { x: { title: { display: true, text: "Importance" } } },
      },
    });
  } catch (e) {
    console.error("Feature importance load error", e);
  }
}

// ----------------- Prediction -----------------
function setupPredict() {
  const checkBtn = document.getElementById("checkBtn");
  checkBtn.addEventListener("click", async () => {
    const url = document.getElementById("urlInput").value.trim();
    if (!url) {
      alert("Enter URL.");
      return;
    }

    const resultDiv = document.getElementById("result");
    const bdWarningDiv = document.getElementById("bdWarning");
    const probLegitDiv = document.getElementById("probLegit");
    const probPhishDiv = document.getElementById("probPhish");

    resultDiv.innerHTML = "Checking...";
    bdWarningDiv.style.display = "none";
    probLegitDiv.style.width = probPhishDiv.style.width = "0%";
    probLegitDiv.innerHTML = probPhishDiv.innerHTML = "";

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      const data = await response.json();
      if (data.error) {
        resultDiv.innerHTML = data.error;
        return;
      }

      const legitProb = (data.probabilities.legitimate * 100).toFixed(1);
      const phishProb = (data.probabilities.phishing * 100).toFixed(1);

      resultDiv.className =
        "result " + (data.prediction === 0 ? "legitimate" : "phishing");
      resultDiv.innerHTML =
        data.prediction === 0 ? "Legitimate URL ✅" : "Phishing URL ⚠️";

      probLegitDiv.style.width = legitProb + "%";
      probLegitDiv.innerHTML = legitProb + "%";
      probPhishDiv.style.width = phishProb + "%";
      probPhishDiv.innerHTML = phishProb + "%";

      if (
        url.toLowerCase().includes(".bd") ||
        /ebl|bracbank|dbbl|ific|citybank|primebank|islami bank/.test(
          url.toLowerCase()
        )
      ) {
        bdWarningDiv.style.display = "block";
        bdWarningDiv.innerHTML =
          "⚠️ This URL is related to a BD domain/bank. Be extra cautious!";
      }
    } catch (err) {
      resultDiv.innerHTML = "Error connecting to API.";
      console.error(err);
    }
  });
}
