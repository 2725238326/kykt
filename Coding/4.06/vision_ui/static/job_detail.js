function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

let elapsedInterval = null;

function startElapsedTimer() {
  const page = document.querySelector("[data-job-created]");
  const badge = document.getElementById("elapsed-badge");
  if (!page || !badge) return;

  const created = new Date(page.dataset.jobCreated);
  if (Number.isNaN(created.getTime())) {
    badge.textContent = "Time --:--";
    return;
  }

  function update() {
    const status = page.dataset.jobStatus;
    const now = new Date();
    const diff = Math.max(0, Math.floor((now - created) / 1000));
    const hours = Math.floor(diff / 3600);
    const minutes = Math.floor((diff % 3600) / 60);
    const seconds = diff % 60;

    const parts = [];
    if (hours > 0) parts.push(String(hours).padStart(2, "0"));
    parts.push(String(minutes).padStart(2, "0"));
    parts.push(String(seconds).padStart(2, "0"));
    badge.textContent = `Time ${parts.join(":")}`;

    if (status === "running") {
      badge.classList.add("is-running");
    } else {
      badge.classList.remove("is-running");
    }

    if (status === "finished" || status === "failed") {
      clearInterval(elapsedInterval);
    }
  }

  update();
  elapsedInterval = setInterval(update, 1000);
}

function renderOutputs(outputs) {
  const grid = document.getElementById("outputs-grid");
  if (!grid) return;

  if (!outputs.length) {
    grid.innerHTML = '<p class="empty" id="outputs-empty">No outputs yet. They will appear here when the remote job finishes.</p>';
    return;
  }

  grid.innerHTML = outputs.map((item, index) => {
    let body;
    if (item.is_image) {
      body = `<img src="${item.url}" alt="${escapeHtml(item.display_name)}" loading="lazy">`;
    } else if (item.is_pointcloud) {
      body = `
        <div class="viewer-container" id="viewer-${index + 1}" data-ply-url="${item.url}">
          <div class="viewer-loading" id="viewer-loading-${index + 1}">
            <div class="spinner"></div>
            Loading point cloud...
          </div>
        </div>
        <div class="viewer-controls">
          <button class="viewer-btn" onclick="resetViewer(${index + 1})" type="button">Reset View</button>
          <a href="${item.url}" download class="viewer-btn viewer-link">Download .ply</a>
        </div>
      `;
    } else {
      body = `<p><a class="inline-link" href="${item.url}" target="_blank">Open / Download</a></p>`;
    }

    return `
      <div class="preview-card">
        <p><strong>${escapeHtml(item.display_name)}</strong></p>
        <p class="meta-line"><code>${escapeHtml(item.relative_path)}</code></p>
        ${body}
      </div>
    `;
  }).join("");

  if (typeof initAllViewers === "function") {
    window.setTimeout(initAllViewers, 120);
  }
}

function renderLogs(logs) {
  const grid = document.getElementById("log-grid");
  if (!grid) return;

  if (!logs.length) {
    grid.innerHTML = '<p class="empty" id="logs-empty">No local log cache yet. Live logs will appear here once the remote task starts.</p>';
    return;
  }

  grid.innerHTML = logs.map((item) => `
    <div class="preview-card">
      <p><strong>${escapeHtml(item.name)}</strong></p>
      <p class="meta-line"><code>${escapeHtml(item.relative_path)}</code></p>
      <pre class="log-box">${escapeHtml(item.tail || "Log file exists but is currently empty.")}</pre>
    </div>
  `).join("");

  grid.querySelectorAll(".log-box").forEach((box) => {
    box.scrollTop = box.scrollHeight;
  });
}

function renderProgress(phaseDisplay) {
  const fill = document.getElementById("job-progress-fill");
  const percent = document.getElementById("job-progress-percent");
  const text = document.getElementById("job-progress-text");
  const phaseText = document.getElementById("job-phase-text");

  if (fill) fill.style.width = `${phaseDisplay.percent}%`;
  if (percent) percent.textContent = `${phaseDisplay.percent}%`;
  if (text) text.textContent = phaseDisplay.description;
  if (phaseText) phaseText.textContent = phaseDisplay.label;

  const steps = document.querySelector(".progress-steps");
  if (!steps) return;

  steps.innerHTML = phaseDisplay.steps.map((item, index) => `
    <div class="progress-step ${item.state === "done" ? "is-done" : ""} ${item.state === "current" ? "is-current" : ""}">
      <span class="step-index">${index + 1}</span>
      <div>
        <strong>${escapeHtml(item.label)}</strong>
        <p>${escapeHtml(item.hint)}</p>
      </div>
    </div>
  `).join("");
}

async function refreshJobState() {
  const page = document.querySelector("[data-job-id]");
  if (!page) return;
  const jobId = page.getAttribute("data-job-id");

  try {
    const response = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
    if (!response.ok) throw new Error("Failed to load job state");

    const data = await response.json();
    const job = data.job;

    page.dataset.jobStatus = job.status;
    document.getElementById("job-status").textContent = job.status;
    document.getElementById("job-phase").textContent = job.phase;
    document.getElementById("job-latest-progress").textContent = job.progress_message || "Waiting for the first remote log line.";

    const pill = document.getElementById("job-status-pill");
    if (pill) {
      pill.textContent = job.status;
      pill.className = `status-pill status-${job.status}`;
    }

    const errorBox = document.getElementById("error-box");
    const errorMessage = document.getElementById("job-error-message");
    if (job.error_message) {
      errorBox?.classList.remove("hidden");
      if (errorMessage) errorMessage.textContent = job.error_message;
    } else {
      errorBox?.classList.add("hidden");
      if (errorMessage) errorMessage.textContent = "";
    }

    const runButton = document.getElementById("run-job-button");
    const retryButton = document.getElementById("retry-job-button");
    if (runButton) runButton.disabled = job.status === "running";
    if (retryButton) retryButton.disabled = job.status === "running";

    renderProgress(data.phase_display);
    renderOutputs(data.outputs || []);
    renderLogs(data.logs || []);
  } catch (_error) {
    // Keep current content and try again soon.
  } finally {
    window.setTimeout(refreshJobState, 2500);
  }
}

window.addEventListener("load", () => {
  startElapsedTimer();
  refreshJobState();
});
