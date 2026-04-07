/* ===== KYKT Vision UI - PLY Point Cloud Viewer (Three.js) ===== */

/**
 * Lightweight PLY viewer using Three.js.
 * Finds all .viewer-container elements with data-ply-url and renders an interactive
 * point cloud with orbit controls inside each container.
 */

const viewers = {};
window.viewers = viewers;
const MAX_PREVIEW_POINTS = 80000;

function parsePLY(text, maxPreviewPoints = MAX_PREVIEW_POINTS) {
  const lines = text.split("\n");
  let vertexCount = 0;
  let headerEnd = 0;
  let hasColor = false;
  const properties = [];

  // Parse header
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line === "end_header") {
      headerEnd = i + 1;
      break;
    }
    if (line.startsWith("element vertex")) {
      vertexCount = parseInt(line.split(" ")[2]);
    }
    if (line.startsWith("property")) {
      const parts = line.split(" ");
      properties.push(parts[parts.length - 1]);
    }
    if (line.includes("red") || line.includes("diffuse_red")) {
      hasColor = true;
    }
  }

  const xIdx = properties.indexOf("x");
  const yIdx = properties.indexOf("y");
  const zIdx = properties.indexOf("z");

  let rIdx = properties.indexOf("red");
  if (rIdx === -1) rIdx = properties.indexOf("diffuse_red");
  let gIdx = properties.indexOf("green");
  if (gIdx === -1) gIdx = properties.indexOf("diffuse_green");
  let bIdx = properties.indexOf("blue");
  if (bIdx === -1) bIdx = properties.indexOf("diffuse_blue");

  hasColor = rIdx !== -1 && gIdx !== -1 && bIdx !== -1;

  const sampleStep = Math.max(1, Math.ceil(vertexCount / maxPreviewPoints));
  const previewCount = Math.ceil(vertexCount / sampleStep);
  const positions = new Float32Array(previewCount * 3);
  const colors = new Float32Array(previewCount * 3);

  // Parse vertices
  let outIdx = 0;
  for (let i = 0; i < vertexCount; i++) {
    if (i % sampleStep !== 0) continue;
    const lineIdx = headerEnd + i;
    if (lineIdx >= lines.length) break;

    const parts = lines[lineIdx].trim().split(/\s+/);
    positions[outIdx * 3] = parseFloat(parts[xIdx]) || 0;
    positions[outIdx * 3 + 1] = parseFloat(parts[yIdx]) || 0;
    positions[outIdx * 3 + 2] = parseFloat(parts[zIdx]) || 0;

    if (hasColor) {
      colors[outIdx * 3] = (parseFloat(parts[rIdx]) || 0) / 255;
      colors[outIdx * 3 + 1] = (parseFloat(parts[gIdx]) || 0) / 255;
      colors[outIdx * 3 + 2] = (parseFloat(parts[bIdx]) || 0) / 255;
    } else {
      // Default gradient color based on Y position
      colors[outIdx * 3] = 0.13;
      colors[outIdx * 3 + 1] = 0.83;
      colors[outIdx * 3 + 2] = 0.93;
    }
    outIdx += 1;
  }

  return {
    positions: positions.slice(0, outIdx * 3),
    colors: colors.slice(0, outIdx * 3),
    vertexCount,
    previewCount: outIdx,
    sampleStep,
  };
}

function createViewer(container, plyUrl) {
  const id = container.id;
  if (viewers[id]) return; // Already initialized

  const width = Math.max(container.clientWidth, 320);
  const height = Math.max(container.clientHeight, 240);

  // Scene
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x050810);

  // Camera
  const camera = new THREE.PerspectiveCamera(60, width / height, 0.01, 1000);
  camera.position.set(0, 0, 3);

  // Renderer
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  // Controls
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.rotateSpeed = 0.8;
  controls.zoomSpeed = 1.2;

  // Store viewer
  viewers[id] = { scene, camera, renderer, controls, container, active: true, frameId: null };

  // Load PLY
  fetch(plyUrl)
    .then(res => {
      if (!res.ok) throw new Error(`PLY 加载失败：${res.status}`);
      return res.text();
    })
    .then(text => {
      const { positions, colors, vertexCount, previewCount, sampleStep } = parsePLY(text);

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

      const material = new THREE.PointsMaterial({
        size: 0.015,
        vertexColors: true,
        sizeAttenuation: true,
      });

      const points = new THREE.Points(geometry, material);

      // Center the point cloud
      geometry.computeBoundingBox();
      const box = geometry.boundingBox;
      const center = new THREE.Vector3();
      box.getCenter(center);
      points.position.sub(center);

      // Scale to fit
      const maxDim = Math.max(
        box.max.x - box.min.x,
        box.max.y - box.min.y,
        box.max.z - box.min.z
      );
      if (maxDim > 0) {
        const scale = 2.5 / maxDim;
        points.scale.set(scale, scale, scale);
      }

      scene.add(points);
      viewers[id].points = points;
      viewers[id].initialCameraPos = camera.position.clone();
      viewers[id].initialTarget = controls.target.clone();

      // Hide loading
      const loading = container.querySelector(".viewer-loading");
      if (loading) {
        loading.innerHTML = `已加载 ${previewCount.toLocaleString()} / ${vertexCount.toLocaleString()} 个点${sampleStep > 1 ? "（预览已下采样）" : ""}`;
        window.setTimeout(() => {
          loading.style.display = "none";
        }, 450);
      }
    })
    .catch(err => {
      const loading = container.querySelector(".viewer-loading");
      if (loading) loading.innerHTML = `<span style="color:var(--danger);">错误：${err.message}</span>`;
    });

  // Animation loop
  function animate() {
    if (!viewers[id]?.active) return;
    viewers[id].frameId = requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  // Resize observer
  const observer = new ResizeObserver(() => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });
  observer.observe(container);
}

function destroyViewer(id) {
  const v = viewers[id];
  if (!v) return;
  v.active = false;
  if (v.frameId) cancelAnimationFrame(v.frameId);
  if (v.points) {
    v.points.geometry?.dispose?.();
    v.points.material?.dispose?.();
    v.scene?.remove?.(v.points);
  }
  v.controls?.dispose?.();
  v.renderer?.dispose?.();
  v.renderer?.domElement?.remove?.();
  delete viewers[id];
}

function resetViewer(index) {
  const id = `viewer-${index}`;
  resetViewerById(id);
}

function resetViewerById(id) {
  const v = viewers[id];
  if (!v) return;

  v.camera.position.copy(v.initialCameraPos || new THREE.Vector3(0, 0, 3));
  v.controls.target.copy(v.initialTarget || new THREE.Vector3(0, 0, 0));
  v.controls.update();
}

function initAllViewers() {
  document.querySelectorAll(".viewer-container[data-ply-url]").forEach(container => {
    const url = container.dataset.plyUrl;
    if (url && typeof THREE !== "undefined") {
      createViewer(container, url);
    }
  });
}
window.destroyViewer = destroyViewer;
window.createViewer = createViewer;
window.resetViewerById = resetViewerById;
