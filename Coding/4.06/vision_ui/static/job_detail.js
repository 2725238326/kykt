function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function statusLabel(status) {
  const labels = {
    created: "已创建",
    ready: "已就绪",
    running: "运行中",
    finished: "已完成",
    failed: "失败",
    cancelled: "已取消",
  };
  return labels[status] || status || "未知";
}

let elapsedInterval = null;
let lastOutputsKey = "";
let lastLogsKey = "";
let refreshTimer = null;
let isRefreshing = false;

function renderResultSummary(summary) {
  const box = document.getElementById("result-summary-box");
  const content = document.getElementById("result-summary-content");
  if (!box || !content) return;

  if (!summary) {
    box.classList.add("hidden");
    content.innerHTML = "";
    return;
  }

  box.classList.remove("hidden");

  const summaryCards = [];
  summaryCards.push(`
    <div class="summary-kv">
      <span>输入数量</span>
      <strong>${summary.inputs?.count ?? 0}</strong>
    </div>
  `);
  summaryCards.push(`
    <div class="summary-kv">
      <span>产物数量</span>
      <strong>${(summary.artifacts || []).length}</strong>
    </div>
  `);
  if (summary.scene_meta?.n_pairs !== undefined) {
    summaryCards.push(`
      <div class="summary-kv">
        <span>图像配对数</span>
        <strong>${summary.scene_meta.n_pairs}</strong>
      </div>
    `);
  }
  if (summary.scene_meta?.n_points !== undefined) {
    summaryCards.push(`
      <div class="summary-kv">
        <span>点云点数</span>
        <strong>${summary.scene_meta.n_points}</strong>
      </div>
    `);
  }

  const blocks = [];
  if ((summary.highlights || []).length) {
    blocks.push(`
      <div class="summary-list-block">
        <strong>关键结果</strong>
        <ul>
          ${(summary.highlights || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
        </ul>
      </div>
    `);
  }
  if ((summary.next_actions || []).length) {
    blocks.push(`
      <div class="summary-list-block">
        <strong>建议后续动作</strong>
        <ul>
          ${(summary.next_actions || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
        </ul>
      </div>
    `);
  }

  content.innerHTML = `
    <div class="result-summary-grid">
      ${summaryCards.join("")}
    </div>
    ${blocks.join("")}
  `;
}

function startElapsedTimer() {
  const page = document.querySelector("[data-job-created]");
  const badge = document.getElementById("elapsed-badge");
  if (!page || !badge) return;

  const created = new Date(page.dataset.jobCreated);
  if (Number.isNaN(created.getTime())) {
    badge.textContent = "耗时 --:--";
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
    badge.textContent = `耗时 ${parts.join(":")}`;

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
    grid.innerHTML = '<p class="empty" id="outputs-empty">还没有输出结果。远端任务完成后，匹配图、点云和日志会显示在这里。</p>';
    return;
  }

  grid.innerHTML = outputs.map((item, index) => {
    let body;
    if (item.is_image) {
      body = `<img src="${item.url}" alt="${escapeHtml(item.display_name)}" loading="lazy">`;
    } else if (item.is_pointcloud) {
      body = `
        <div class="pointcloud-placeholder">
          <span class="pointcloud-icon">PLY</span>
          <div>
            <strong>点云文件已就绪</strong>
            <p>浏览器点云预览已关闭，避免页面卡顿。建议用 MeshLab 打开，或直接下载 .ply 文件。</p>
          </div>
        </div>
        <div class="viewer-controls">
          <button class="viewer-btn js-open-local-output" data-relative-path="${escapeHtml(item.relative_path)}" type="button">用 MeshLab 打开</button>
          <a href="${item.url}" download class="viewer-btn viewer-link">下载 .ply</a>
        </div>
      `;
    } else {
      body = `<p><a class="inline-link" href="${item.url}" target="_blank">打开或下载</a></p>`;
    }

    return `
      <div class="preview-card">
        <p><strong>${escapeHtml(item.display_name)}</strong></p>
        <p class="meta-line"><code>${escapeHtml(item.relative_path)}</code></p>
        ${body}
      </div>
    `;
  }).join("");

  bindOutputActions();
}

function stableKey(items) {
  return JSON.stringify(
    (items || []).map((item) => [
      item.relative_path,
      item.url,
      item.display_name,
      item.tail,
    ])
  );
}

function bindOutputActions() {
  document.querySelectorAll(".js-open-local-output").forEach((button) => {
    button.onclick = () => {
      openLocalOutput(button.dataset.relativePath || "");
    };
  });
}

async function openLocalOutput(relativePath) {
  const page = document.querySelector("[data-job-id]");
  if (!page || !relativePath) return;
  const jobId = page.getAttribute("data-job-id");
  const formData = new FormData();
  formData.append("relative_path", relativePath);

  try {
    const response = await fetch(`/jobs/${jobId}/open-output`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || "打开本地文件失败");
    }
  } catch (error) {
    window.alert(`本地打开失败：${error.message || error}`);
  }
}

function renderLogs(logs) {
  const grid = document.getElementById("log-grid");
  if (!grid) return;

  if (!logs.length) {
    grid.innerHTML = '<p class="empty" id="logs-empty">还没有本地日志缓存。远端任务开始后，日志会自动显示在这里。</p>';
    return;
  }

  grid.innerHTML = logs.map((item) => `
    <div class="preview-card">
      <p><strong>${escapeHtml(item.name)}</strong></p>
      <p class="meta-line"><code>${escapeHtml(item.relative_path)}</code></p>
      <pre class="log-box">${escapeHtml(item.tail || "日志文件存在，但当前还是空的。")}</pre>
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
  if (isRefreshing) return;
  const page = document.querySelector("[data-job-id]");
  if (!page) return;
  const jobId = page.getAttribute("data-job-id");
  isRefreshing = true;

  try {
    const response = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
    if (!response.ok) throw new Error("读取任务状态失败");

    const data = await response.json();
    const job = data.job;

    page.dataset.jobStatus = job.status;
    document.getElementById("job-status").textContent = statusLabel(job.status);
    document.getElementById("job-phase").textContent = data.phase_display.label;
    document.getElementById("job-latest-progress").textContent = job.progress_message || "正在等待第一条远端日志。";

    const pill = document.getElementById("job-status-pill");
    if (pill) {
      pill.textContent = statusLabel(job.status);
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
    const cancelButton = document.getElementById("cancel-job-button");
    if (runButton) runButton.disabled = job.status === "running";
    if (retryButton) retryButton.disabled = job.status === "running";
    if (cancelButton) cancelButton.disabled = job.status !== "running";

    renderProgress(data.phase_display);
    const outputs = data.outputs || [];
    const logs = data.logs || [];
    renderResultSummary(data.result_summary || null);
    const outputsKey = stableKey(outputs);
    const logsKey = stableKey(logs);

    if (outputsKey !== lastOutputsKey) {
      lastOutputsKey = outputsKey;
      renderOutputs(outputs);
    }
    if (logsKey !== lastLogsKey) {
      lastLogsKey = logsKey;
      renderLogs(logs);
    }
  } catch (_error) {
    // Keep current content and try again soon.
  } finally {
    isRefreshing = false;
    const pageStatus = page?.dataset?.jobStatus;
    const delay = pageStatus === "running" ? 2500 : 10000;
    refreshTimer = window.setTimeout(refreshJobState, delay);
  }
}

window.addEventListener("load", () => {
  startElapsedTimer();
  bindOutputActions();
  if (refreshTimer) window.clearTimeout(refreshTimer);
  refreshJobState();
});
