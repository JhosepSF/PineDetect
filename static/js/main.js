/**
 * PineDetect - Lógica Interactiva del Cliente (JavaScript)
 * Manejo de Drag & Drop, AJAX, renderizado dinámico, Chart.js y descargas
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elementos DOM
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const previewContainer = document.getElementById('previewContainer');
  const previewImg = document.getElementById('previewImg');
  const previewFilename = document.getElementById('previewFilename');
  const previewDimensions = document.getElementById('previewDimensions');
  const previewSize = document.getElementById('previewSize');
  const resetFileBtn = document.getElementById('resetFileBtn');
  const analyzeBtn = document.getElementById('analyzeBtn');

  // Sliders
  const confSlider = document.getElementById('confSlider');
  const confValue = document.getElementById('confValue');
  const iouSlider = document.getElementById('iouSlider');
  const iouValue = document.getElementById('iouValue');

  // Overlays y alertas
  const loadingOverlay = document.getElementById('loadingOverlay');
  const errorAlert = document.getElementById('errorAlert');
  const errorMessage = document.getElementById('errorMessage');
  const guideSection = document.getElementById('guideSection');
  const resultsSection = document.getElementById('resultsSection');
  const noDetectionsAlert = document.getElementById('noDetectionsAlert');
  const analyticsSection = document.getElementById('analyticsSection');

  // Resultados DOM
  const originalResultImg = document.getElementById('originalResultImg');
  const annotatedResultImg = document.getElementById('annotatedResultImg');
  const metricTotal = document.getElementById('metricTotal');
  const metricAvgConf = document.getElementById('metricAvgConf');
  const metricTime = document.getElementById('metricTime');
  const metricPredominant = document.getElementById('metricPredominant');

  const countInmadura = document.getElementById('countInmadura');
  const countMadura = document.getElementById('countMadura');
  const countSobremadura = document.getElementById('countSobremadura');

  const detectionsTableBody = document.getElementById('detectionsTableBody');

  // Botones de descarga
  const downloadPngBtn = document.getElementById('downloadPngBtn');
  const downloadCsvBtn = document.getElementById('downloadCsvBtn');
  const downloadJsonBtn = document.getElementById('downloadJsonBtn');

  // Estado local
  let currentFile = null;
  let currentAnalysisData = null;
  let maturityChartInstance = null;

  // Obtener CSRF Token de cookies de Django
  function getCsrfToken() {
    const name = 'csrftoken';
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
      const cookies = document.cookie.split(';');
      for (let i = 0; i < cookies.length; i++) {
        const cookie = cookies[i].trim();
        if (cookie.substring(0, name.length + 1) === (name + '=')) {
          cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
          break;
        }
      }
    }
    return cookieValue || document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';
  }

  // =========================================================================
  // Control de Sliders
  // =========================================================================
  if (confSlider && confValue) {
    confSlider.addEventListener('input', (e) => {
      confValue.textContent = parseFloat(e.target.value).toFixed(2);
    });
  }

  if (iouSlider && iouValue) {
    iouSlider.addEventListener('input', (e) => {
      iouValue.textContent = parseFloat(e.target.value).toFixed(2);
    });
  }

  // =========================================================================
  // Drag and Drop & Selección de Archivos
  // =========================================================================
  if (dropzone && fileInput) {
    dropzone.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('dragover');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        handleSelectedFile(files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleSelectedFile(e.target.files[0]);
      }
    });
  }

  if (resetFileBtn) {
    resetFileBtn.addEventListener('click', () => {
      clearSelectedFile();
    });
  }

  function showError(msg) {
    if (errorMessage && errorAlert) {
      errorMessage.textContent = msg;
      errorAlert.style.display = 'flex';
      errorAlert.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  function hideError() {
    if (errorAlert) {
      errorAlert.style.display = 'none';
    }
  }

  function handleSelectedFile(file) {
    hideError();
    const validExtensions = ['.jpg', '.jpeg', '.png', '.webp'];
    const extension = '.' + file.name.split('.').pop().toLowerCase();

    if (!validExtensions.includes(extension)) {
      showError(`Formato no permitido (${extension}). Solo se admiten archivos JPG, JPEG, PNG y WEBP.`);
      return;
    }

    const maxBytes = 10 * 1024 * 1024; // 10 MB
    if (file.size > maxBytes) {
      const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
      showError(`El archivo (${sizeMb} MB) supera el tamaño máximo permitido de 10 MB.`);
      return;
    }

    currentFile = file;

    // Crear vista previa con FileReader
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        previewImg.src = e.target.result;
        previewFilename.textContent = file.name;
        previewDimensions.textContent = `${img.naturalWidth} × ${img.naturalHeight} px`;
        previewSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;

        dropzone.style.display = 'none';
        previewContainer.style.display = 'flex';
        guideSection.style.display = 'none';
        resultsSection.style.display = 'none';
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  function clearSelectedFile() {
    currentFile = null;
    currentAnalysisData = null;
    if (fileInput) fileInput.value = '';
    if (previewImg) previewImg.src = '';
    if (dropzone) dropzone.style.display = 'flex';
    if (previewContainer) previewContainer.style.display = 'none';
    if (guideSection) guideSection.style.display = 'grid';
    if (resultsSection) resultsSection.style.display = 'none';
    hideError();
  }

  // =========================================================================
  // Ejecución de Inferencia AJAX
  // =========================================================================
  if (analyzeBtn) {
    analyzeBtn.addEventListener('click', async () => {
      if (!currentFile) {
        showError('Por favor selecciona una imagen antes de analizar.');
        return;
      }

      hideError();
      loadingOverlay.style.display = 'block';
      resultsSection.style.display = 'none';
      analyzeBtn.disabled = true;

      let rawConf = confSlider ? confSlider.value : '0.25';
      let confNum = parseFloat(String(rawConf).replace(',', '.'));
      if (isNaN(confNum) || confNum <= 0) confNum = 0.25;
      if (confNum > 1.0 && confNum <= 100.0) confNum = confNum / 100.0;

      let rawIou = iouSlider ? iouSlider.value : '0.70';
      let iouNum = parseFloat(String(rawIou).replace(',', '.'));
      if (isNaN(iouNum) || iouNum <= 0) iouNum = 0.70;
      if (iouNum > 1.0 && iouNum <= 100.0) iouNum = iouNum / 100.0;

      const formData = new FormData();
      formData.append('image', currentFile);
      formData.append('confidence', confNum.toFixed(2));
      formData.append('iou', iouNum.toFixed(2));

      try {
        const response = await fetch('/api/analyze/', {
          method: 'POST',
          headers: {
            'X-CSRFToken': getCsrfToken()
          },
          body: formData
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
          throw new Error(data.error || 'Error al procesar la imagen.');
        }

        currentAnalysisData = data;
        renderResults(data);
      } catch (err) {
        console.error('Error durante la inferencia:', err);
        showError(err.message || 'Error de conexión con el servidor PineDetect.');
      } finally {
        loadingOverlay.style.display = 'none';
        analyzeBtn.disabled = false;
      }
    });
  }

  // =========================================================================
  // Renderizado de Resultados
  // =========================================================================
  function renderResults(data) {
    const summary = data.summary;
    const detections = data.detections;
    const tableRows = data.table_rows;

    // Imágenes original y procesada
    originalResultImg.src = data.original_image;
    annotatedResultImg.src = data.annotated_image;

    // Métricas
    metricTotal.textContent = summary.total_detections;
    metricAvgConf.textContent = summary.average_confidence_percentage !== null 
      ? `${summary.average_confidence_percentage.toFixed(1)} %` 
      : 'N/A';
    metricTime.innerHTML = `${summary.inference_time_ms.toFixed(1)} <small style="font-size:1rem;">ms</small>`;
    metricPredominant.textContent = summary.predominant_class;

    // Desglose por clase
    const inmCnt = summary.counts_by_class.inmadura || 0;
    const inmPct = summary.percentages_by_class.inmadura || 0.0;
    const madCnt = summary.counts_by_class.madura || 0;
    const madPct = summary.percentages_by_class.madura || 0.0;
    const sobCnt = summary.counts_by_class.sobremadura || 0;
    const sobPct = summary.percentages_by_class.sobremadura || 0.0;

    countInmadura.textContent = `${inmCnt} (${inmPct.toFixed(1)}%)`;
    countMadura.textContent = `${madCnt} (${madPct.toFixed(1)}%)`;
    countSobremadura.textContent = `${sobCnt} (${sobPct.toFixed(1)}%)`;

    // Manejo de caso sin detecciones
    if (summary.total_detections === 0) {
      noDetectionsAlert.style.display = 'block';
      analyticsSection.style.display = 'none';
      detectionsTableBody.innerHTML = '<tr><td colspan="9" style="text-align:center; color:#64748B;">No hay detecciones registradas.</td></tr>';
    } else {
      noDetectionsAlert.style.display = 'none';
      analyticsSection.style.display = 'grid';

      // Actualizar Gráfico Chart.js
      renderMaturityChart(summary.counts_by_class, summary.percentages_by_class, summary.total_detections);

      // Llenar tabla de detecciones
      renderTable(tableRows);
    }

    resultsSection.style.display = 'flex';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function renderMaturityChart(counts, percentages, total) {
    const ctx = document.getElementById('maturityChart');
    if (!ctx) return;

    if (maturityChartInstance) {
      maturityChartInstance.destroy();
    }

    const categories = ['Inmadura', 'Madura', 'Sobremadura'];
    const values = [
      counts.inmadura || 0,
      counts.madura || 0,
      counts.sobremadura || 0
    ];
    const bgColors = ['#2CA02C', '#F2B134', '#D62728'];

    maturityChartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: categories,
        datasets: [{
          label: 'Cantidad de Piñas',
          data: values,
          backgroundColor: bgColors,
          borderColor: 'rgba(0,0,0,0.1)',
          borderWidth: 1,
          borderRadius: 8,
          barPercentage: 0.65
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              afterLabel: function(context) {
                const count = context.parsed.y;
                const pct = total > 0 ? ((count / total) * 100).toFixed(1) : '0.0';
                return `Proporción: ${pct}%`;
              }
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            ticks: {
              stepSize: 1,
              precision: 0,
              font: { family: 'inherit', size: 12 }
            },
            grid: { color: '#E2E8F0' }
          },
          x: {
            grid: { display: false },
            ticks: { font: { family: 'inherit', size: 13, weight: 'bold' } }
          }
        }
      }
    });
  }

  function renderTable(rows) {
    detectionsTableBody.innerHTML = '';
    if (!rows || rows.length === 0) {
      detectionsTableBody.innerHTML = '<tr><td colspan="9" style="text-align:center; color:#64748B;">No hay detecciones.</td></tr>';
      return;
    }

    rows.forEach(row => {
      const tr = document.createElement('tr');
      const maturityLower = row['Estado de Madurez'].toLowerCase();

      tr.innerHTML = `
        <td><b>#${row['N° Detección']}</b></td>
        <td><span class="class-badge ${maturityLower}">${row['Estado de Madurez']}</span></td>
        <td><b>${row['Confianza (%)']}</b></td>
        <td>${row['X Mínimo']}</td>
        <td>${row['Y Mínimo']}</td>
        <td>${row['X Máximo']}</td>
        <td>${row['Y Máximo']}</td>
        <td>${row['Ancho (px)']}</td>
        <td>${row['Alto (px)']}</td>
      `;
      detectionsTableBody.appendChild(tr);
    });
  }

  // =========================================================================
  // Descargas de Archivos
  // =========================================================================

  // 1. Descarga de Imagen PNG
  if (downloadPngBtn) {
    downloadPngBtn.addEventListener('click', () => {
      if (!currentAnalysisData || !currentAnalysisData.annotated_image) return;
      const baseName = currentAnalysisData.filename.split('.')[0] || 'pina';
      const a = document.createElement('a');
      a.href = currentAnalysisData.annotated_image;
      a.download = `pinedetect_${baseName}_analizada.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    });
  }

  // 2. Descarga de CSV con BOM
  if (downloadCsvBtn) {
    downloadCsvBtn.addEventListener('click', async () => {
      if (!currentAnalysisData) return;
      const baseName = currentAnalysisData.filename.split('.')[0] || 'pina';

      try {
        const response = await fetch('/api/export/csv/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCsrfToken()
          },
          body: JSON.stringify({
            filename: `pinedetect_${baseName}`,
            detections: currentAnalysisData.detections
          })
        });

        if (!response.ok) throw new Error('Error al descargar CSV.');

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `pinedetect_${baseName}_detecciones.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
      } catch (err) {
        console.error(err);
        showError('No se pudo generar el archivo CSV.');
      }
    });
  }

  // 3. Descarga de JSON Estructurado
  if (downloadJsonBtn) {
    downloadJsonBtn.addEventListener('click', async () => {
      if (!currentAnalysisData) return;
      const baseName = currentAnalysisData.filename.split('.')[0] || 'pina';

      try {
        const response = await fetch('/api/export/json/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCsrfToken()
          },
          body: JSON.stringify({
            filename: `pinedetect_${baseName}`,
            detections: currentAnalysisData.detections,
            summary: currentAnalysisData.summary,
            thresholds: currentAnalysisData.thresholds
          })
        });

        if (!response.ok) throw new Error('Error al descargar JSON.');

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `pinedetect_${baseName}_resumen.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
      } catch (err) {
        console.error(err);
        showError('No se pudo generar el archivo JSON.');
      }
    });
  }
});
