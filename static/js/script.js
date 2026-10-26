document.addEventListener("DOMContentLoaded", () => {
  const menuToggle = document.getElementById("menuToggle");
  const sidebar = document.getElementById("sidebar");
  const sidebarScrim = document.getElementById("sidebarScrim");
  const closeSidebar = () => {
    sidebar?.classList.remove("open");
    sidebarScrim?.classList.remove("open");
  };

  menuToggle?.addEventListener("click", () => {
    sidebar?.classList.toggle("open");
    sidebarScrim?.classList.toggle("open");
  });
  sidebarScrim?.addEventListener("click", closeSidebar);

  const form = document.getElementById("inspectionForm");
  if (!form) return;

  const input = document.getElementById("imageInput");
  const dropZone = document.getElementById("dropZone");
  const placeholder = document.getElementById("dropPlaceholder");
  const preview = document.getElementById("imagePreview");
  const previewChange = document.getElementById("previewChange");
  const fileMeta = document.getElementById("fileMeta");
  const fileName = document.getElementById("fileName");
  const removeFile = document.getElementById("removeFile");
  const errorBox = document.getElementById("formError");
  const submitButton = document.getElementById("analyzeButton");
  const liveResult = document.getElementById("liveResult");
  let previewUrl = null;

  const showSelectedFile = (file) => {
    errorBox.hidden = true;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    preview.hidden = false;
    placeholder.hidden = true;
    previewChange.hidden = false;
    fileMeta.hidden = false;
    fileName.textContent = `${file.name} · ${(file.size / (1024 * 1024)).toFixed(2)} MB`;
  };

  input.addEventListener("change", () => {
    const file = input.files[0];
    if (file) showSelectedFile(file);
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.remove("dragover");
    });
  });
  dropZone.addEventListener("drop", (event) => {
    const [file] = event.dataTransfer.files;
    if (!file) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    input.files = transfer.files;
    showSelectedFile(file);
  });

  removeFile.addEventListener("click", () => {
    input.value = "";
    preview.removeAttribute("src");
    preview.hidden = true;
    placeholder.hidden = false;
    previewChange.hidden = true;
    fileMeta.hidden = true;
    liveResult.hidden = true;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    errorBox.hidden = true;
    if (!input.files[0]) {
      errorBox.textContent = "Choose an image file before analyzing the belt.";
      errorBox.hidden = false;
      return;
    }

    submitButton.disabled = true;
    submitButton.querySelector("span").textContent = "Analyzing image...";
    try {
      const response = await fetch(form.action, {
        method: "POST",
        body: new FormData(form),
        headers: { Accept: "application/json" },
      });
      const payload = await response.json();
      if (!response.ok || !payload.success) throw new Error(payload.error || "The image could not be analyzed.");
      renderLiveResult(payload);
      liveResult.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
      errorBox.textContent = error.message || "The image could not be analyzed. Please try again.";
      errorBox.hidden = false;
    } finally {
      submitButton.disabled = false;
      submitButton.querySelector("span").textContent = "Analyze belt";
    }
  });

  function renderLiveResult(payload) {
    const record = payload.record;
    liveResult.replaceChildren();
    const heading = document.createElement("div");
    heading.className = "panel-heading";
    const title = document.createElement("div");
    title.innerHTML = '<div class="section-kicker">INSPECTION COMPLETE</div><h2>Detection results</h2>';
    const link = document.createElement("a");
    link.className = "text-link";
    link.href = payload.results_url;
    link.textContent = "Open full report ";
    const arrow = document.createElement("i");
    arrow.className = "bi bi-arrow-right";
    link.append(arrow);
    heading.append(title, link);

    const content = document.createElement("div");
    content.className = "live-result-content";
    const image = document.createElement("img");
    image.className = "live-result-image";
    image.src = payload.annotated_image;
    image.alt = "Annotated conveyor belt inspection image";
    const details = document.createElement("div");
    details.className = "live-result-details";
    const status = document.createElement("div");
    status.className = "live-detection";
    const riskLabel = record.risk_score === null ? "Risk not assessed" : `${record.risk_level} · ${record.risk_score}%`;
    status.innerHTML = `<strong>${record.defect_count} damage${record.defect_count === 1 ? "" : "s"} · ${riskLabel}</strong>`;
    const recommendation = document.createElement("small");
    recommendation.textContent = record.recommendation;
    status.append(recommendation);
    details.append(status);

    record.detections.forEach((detection) => {
      const row = document.createElement("div");
      row.className = "live-detection";
      const name = document.createElement("strong");
      name.textContent = detection.class_name;
      const meta = document.createElement("small");
      meta.textContent = `${detection.confidence_percent}% confidence · box ${detection.bbox.join(", ")}${detection.is_damage ? " · damage" : " · context"}`;
      row.append(name, meta);
      details.append(row);
    });
    content.append(image, details);
    liveResult.append(heading, content);
    liveResult.hidden = false;
  }
});