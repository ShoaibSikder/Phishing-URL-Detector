window.addEventListener("load", async () => {
  const accDiv = document.getElementById("modelAccuracy");
  try {
    const response = await fetch("http://127.0.0.1:8000/metrics");
    const data = await response.json();
    accDiv.innerHTML = data.test_accuracy
      ? `Model Test Accuracy: ${data.test_accuracy}%`
      : "Model Accuracy: N/A";
  } catch (err) {
    accDiv.innerHTML = "Error fetching model accuracy";
    console.error(err);
  }
});

document.getElementById("checkBtn").addEventListener("click", async () => {
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
    const response = await fetch("http://127.0.0.1:8000/predict", {
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
