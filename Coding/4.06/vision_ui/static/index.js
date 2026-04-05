function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function initDropZone() {
  const zone = document.getElementById("drop-zone");
  const input = document.getElementById("file-input");
  const list = document.getElementById("file-list");
  if (!zone || !input) return;

  const dataTransfer = new DataTransfer();

  function syncFiles() {
    input.files = dataTransfer.files;
    renderFileList();
  }

  function renderFileList() {
    if (!list) return;
    const files = Array.from(dataTransfer.files);
    if (!files.length) {
      list.innerHTML = "";
      return;
    }

    list.innerHTML = files.map((file, index) => {
      const size = file.size < 1024 * 1024
        ? `${(file.size / 1024).toFixed(1)} KB`
        : `${(file.size / (1024 * 1024)).toFixed(1)} MB`;

      return `
        <span class="file-chip">
          <span>${escapeHtml(file.name)}</span>
          <span style="opacity:0.55;">(${size})</span>
          <span class="chip-remove" data-index="${index}" title="Remove">x</span>
        </span>
      `;
    }).join("");

    list.querySelectorAll(".chip-remove").forEach((button) => {
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        const index = Number(button.dataset.index);
        const nextTransfer = new DataTransfer();
        Array.from(dataTransfer.files).forEach((file, fileIndex) => {
          if (fileIndex !== index) {
            nextTransfer.items.add(file);
          }
        });
        dataTransfer.items.clear();
        Array.from(nextTransfer.files).forEach((file) => dataTransfer.items.add(file));
        syncFiles();
      });
    });
  }

  ["dragenter", "dragover"].forEach((eventName) => {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.remove("dragover");
    });
  });

  zone.addEventListener("drop", (event) => {
    const files = event.dataTransfer?.files;
    if (!files) return;
    Array.from(files).forEach((file) => dataTransfer.items.add(file));
    syncFiles();
  });

  input.addEventListener("change", () => {
    Array.from(input.files).forEach((file) => dataTransfer.items.add(file));
    syncFiles();
  });
}

function renderJobs(items) {
  const container = document.getElementById("job-list");
  if (!container) return;

  if (!items.length) {
    container.innerHTML = '<p class="empty" id="job-list-empty">No jobs yet. Upload inputs to create the first local job.</p>';
    return;
  }

  container.innerHTML = items.map(({ job, phase_display }) => {
    const badgeClass = job.model === "monst3r" ? "badge badge-monst3r" : "badge";
    return `
      <a class="job-card" href="/jobs/${job.job_id}" id="job-card-${escapeHtml(job.job_id)}">
        <div class="job-head">
          <span class="job-id">${escapeHtml(job.job_id)}</span>
          <span class="${badgeClass}">${escapeHtml(job.model)}</span>
        </div>
        <div class="job-meta">
          <span>Status: ${escapeHtml(job.status)}</span>
          <span>Phase: ${escapeHtml(phase_display.label)}</span>
          <span>Inputs: ${job.input_files.length}</span>
          <span>Outputs: ${job.output_files.length}</span>
        </div>
        <div class="job-mini-progress">
          <div class="status-pill status-${escapeHtml(job.status)}">${escapeHtml(job.status)}</div>
          <div class="mini-track" aria-hidden="true">
            <div class="mini-fill" style="width: ${phase_display.percent}%;"></div>
          </div>
          <span class="mini-percent">${phase_display.percent}%</span>
        </div>
        <p class="job-notes">${escapeHtml(job.progress_message || phase_display.description)}</p>
      </a>
    `;
  }).join("");
}

async function refreshJobs() {
  try {
    const response = await fetch("/api/jobs", { cache: "no-store" });
    if (response.ok) {
      const data = await response.json();
      renderJobs(data.jobs || []);
    }
  } catch (_error) {
    // Keep existing content on transient failures.
  } finally {
    window.setTimeout(refreshJobs, 3000);
  }
}

window.addEventListener("load", () => {
  initDropZone();
  refreshJobs();
});
