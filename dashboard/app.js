const API_BASE_URL = 'https://microscopic-lung-cancer-detection-1.onrender.com';

// ------------------------------------------------------------
// Authentication / UI elements
// ------------------------------------------------------------

const authBackdrop = document.querySelector('#authBackdrop');

function showToast(message) {
  const toast = document.querySelector('#toast');

  if (!toast) {
    console.log(message);
    return;
  }

  toast.textContent = message;
  toast.classList.add('show');

  setTimeout(() => {
    toast.classList.remove('show');
  }, 3500);
}


// ------------------------------------------------------------
// Utility functions
// ------------------------------------------------------------

function getElement(selector) {
  return document.querySelector(selector);
}

function safeText(element, text) {
  if (element) {
    element.textContent = text;
  }
}

function formatPercent(value) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return '—';
  }

  return `${(number * 100).toFixed(2)}%`;
}


// ------------------------------------------------------------
// Authentication
// ------------------------------------------------------------

const loginButton = getElement('#loginButton');
const signupButton = getElement('#signupButton');
const logoutButton = getElement('#logoutButton');

if (loginButton) {
  loginButton.addEventListener('click', () => {
    if (authBackdrop) {
      authBackdrop.classList.add('active');
    }
  });
}

if (signupButton) {
  signupButton.addEventListener('click', () => {
    if (authBackdrop) {
      authBackdrop.classList.add('active');
    }
  });
}

if (logoutButton) {
  logoutButton.addEventListener('click', () => {
    localStorage.removeItem('lungCancerUser');
    showToast('Logged out successfully');
  });
}


// ------------------------------------------------------------
// Close authentication modal
// ------------------------------------------------------------

const authClose = getElement('#authClose');

if (authClose) {
  authClose.addEventListener('click', () => {
    if (authBackdrop) {
      authBackdrop.classList.remove('active');
    }
  });
}

if (authBackdrop) {
  authBackdrop.addEventListener('click', (event) => {
    if (event.target === authBackdrop) {
      authBackdrop.classList.remove('active');
    }
  });
}


// ------------------------------------------------------------
// Navigation
// ------------------------------------------------------------

document.querySelectorAll('[data-scroll]').forEach((element) => {
  element.addEventListener('click', () => {
    const target = element.getAttribute('data-scroll');

    if (!target) {
      return;
    }

    const section = document.querySelector(target);

    if (section) {
      section.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });
    }
  });
});


// ------------------------------------------------------------
// Mobile navigation
// ------------------------------------------------------------

const menuButton = getElement('#menuButton');
const mobileMenu = getElement('#mobileMenu');

if (menuButton && mobileMenu) {
  menuButton.addEventListener('click', () => {
    mobileMenu.classList.toggle('active');
  });
}


// ------------------------------------------------------------
// File upload
// ------------------------------------------------------------

const imageInput = getElement('#imageInput');
const uploadArea = getElement('#uploadArea');
const previewImage = getElement('#previewImage');
const previewContainer = getElement('#previewContainer');

let selectedImage = null;

function displayImage(file) {
  if (!file) {
    return;
  }

  if (!file.type.startsWith('image/')) {
    showToast('Please select a valid image file.');
    return;
  }

  selectedImage = file;

  const reader = new FileReader();

  reader.onload = (event) => {
    const imageData = event.target.result;

    if (previewImage) {
      previewImage.src = imageData;
    }

    if (previewContainer) {
      previewContainer.classList.add('active');
    }
  };

  reader.onerror = () => {
    showToast('Unable to read the selected image.');
  };

  reader.readAsDataURL(file);
}

if (imageInput) {
  imageInput.addEventListener('change', (event) => {
    const file = event.target.files?.[0];

    if (file) {
      displayImage(file);
    }
  });
}


// ------------------------------------------------------------
// Drag and drop
// ------------------------------------------------------------

if (uploadArea) {
  uploadArea.addEventListener('dragover', (event) => {
    event.preventDefault();
    uploadArea.classList.add('dragging');
  });

  uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragging');
  });

  uploadArea.addEventListener('drop', (event) => {
    event.preventDefault();

    uploadArea.classList.remove('dragging');

    const file = event.dataTransfer?.files?.[0];

    if (file) {
      displayImage(file);
    }
  });
}


// ------------------------------------------------------------
// Clear selected image
// ------------------------------------------------------------

const clearImageButton = getElement('#clearImage');

if (clearImageButton) {
  clearImageButton.addEventListener('click', () => {
    selectedImage = null;

    if (imageInput) {
      imageInput.value = '';
    }

    if (previewImage) {
      previewImage.src = '';
    }

    if (previewContainer) {
      previewContainer.classList.remove('active');
    }
  });
}


// ------------------------------------------------------------
// Model selection
// ------------------------------------------------------------

function getSelectedModel() {
  const modelSelect = getElement('#modelSelect');

  if (!modelSelect) {
    return 'fedprox';
  }

  const selection = modelSelect.value || '';

  if (selection.startsWith('FedAvg')) {
    return 'fedavg';
  }

  if (selection.startsWith('Hospital 1')) {
    return 'hospital1';
  }

  if (selection.startsWith('Hospital 2')) {
    return 'hospital2';
  }

  if (selection.startsWith('Hospital 3')) {
    return 'hospital3';
  }

  if (selection.startsWith('FedProx')) {
    return 'fedprox';
  }

  return 'fedprox';
}


// ------------------------------------------------------------
// Live metrics
// ------------------------------------------------------------

function loadLiveMetrics() {
  const status = getElement('#chartLiveStatus');

  safeText(status, 'Loading live metrics...');

  fetch(`${API_BASE_URL}/api/metrics`, {
    method: 'GET',
    headers: {
      'Accept': 'application/json'
    }
  })
    .then(async (response) => {
      const contentType = response.headers.get('content-type') || '';

      if (!response.ok) {
        const text = await response.text();
        throw new Error(
          `Metrics request failed (${response.status}): ${text.slice(0, 200)}`
        );
      }

      if (!contentType.includes('application/json')) {
        const text = await response.text();

        throw new Error(
          `Backend returned non-JSON response: ${text.slice(0, 200)}`
        );
      }

      return response.json();
    })
    .then((data) => {
      if (!data || !Array.isArray(data.metrics)) {
        throw new Error('Invalid metrics response from backend.');
      }

      renderMetrics(data.metrics);

      safeText(status, 'Live metrics loaded');
    })
    .catch((error) => {
      console.error('Metrics error:', error);

      safeText(
        status,
        'Unable to load live metrics'
      );
    });
}


// ------------------------------------------------------------
// Render metrics
// ------------------------------------------------------------

function renderMetrics(metrics) {
  const container =
    getElement('#metricsContainer') ||
    getElement('#metricsGrid') ||
    getElement('#metricsTable');

  if (!container) {
    return;
  }

  /*
   * If the existing HTML already contains a metrics table,
   * update it without destroying the existing design.
   */

  const rows = metrics
    .map((item) => {
      const model = item.model ?? 'Unknown';

      const accuracy = formatPercent(item.accuracy);
      const precision = formatPercent(item.precision);
      const recall = formatPercent(item.recall);
      const macroF1 = formatPercent(item.macroF1);

      return `
        <div class="metric-row">
          <div class="metric-model">${model}</div>
          <div>${accuracy}</div>
          <div>${precision}</div>
          <div>${recall}</div>
          <div>${macroF1}</div>
        </div>
      `;
    })
    .join('');

  /*
   * Only create our fallback metric markup when the target
   * container is clearly intended to hold dynamic metrics.
   */
  if (container.id === 'metricsContainer' || container.id === 'metricsGrid') {
    container.innerHTML = rows;
  }
}


// ------------------------------------------------------------
// Analyze image
// ------------------------------------------------------------

const analyzeButton = getElement('#analyzeButton');

if (analyzeButton) {
  analyzeButton.addEventListener('click', async () => {
    const image = getElement('#previewImage');
    const resultStatus = getElement('#resultStatus');
    const analysisMessage = getElement('#analysisMessage');

    if (!image || !image.src || image.src === window.location.href) {
      showToast('Please upload a microscopic lung image first.');
      return;
    }

    const model = getSelectedModel();

    analyzeButton.disabled = true;

    safeText(
      resultStatus,
      'Running secure backend model'
    );

    safeText(
      analysisMessage,
      'Uploading image and running the selected model...'
    );

    try {
      const response = await fetch(`${API_BASE_URL}/api/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          image: image.src,
          model: model
        })
      });

      const contentType =
        response.headers.get('content-type') || '';

      if (!response.ok) {
        const text = await response.text();

        throw new Error(
          `Prediction request failed (${response.status}): ${text.slice(0, 300)}`
        );
      }

      if (!contentType.includes('application/json')) {
        const text = await response.text();

        throw new Error(
          `Backend returned non-JSON response: ${text.slice(0, 300)}`
        );
      }

      const data = await response.json();

      console.log('Prediction response:', data);

      if (!data) {
        throw new Error('Empty prediction response.');
      }

      /*
       * Update prediction/result elements.
       */

      const predictionElement =
        getElement('#prediction') ||
        getElement('#resultPrediction') ||
        getElement('#diagnosis');

      const confidenceElement =
        getElement('#confidence') ||
        getElement('#resultConfidence');

      const modelElement =
        getElement('#resultModel') ||
        getElement('#modelResult');

      const deviceElement =
        getElement('#resultDevice') ||
        getElement('#deviceResult');

      /*
       * Different backend response field names are supported
       * so the frontend remains compatible with the existing
       * dashboard_server.py response.
       */

      const prediction =
        data.prediction ??
        data.class_name ??
        data.label ??
        data.result ??
        'Unknown';

      const confidence =
        data.confidence ??
        data.probability ??
        data.score ??
        null;

      const modelName =
        data.model ??
        model;

      const device =
        data.device ??
        'backend';

      safeText(predictionElement, prediction);

      if (confidenceElement) {
        if (confidence !== null) {
          const numericConfidence = Number(confidence);

          if (Number.isFinite(numericConfidence)) {
            const displayConfidence =
              numericConfidence <= 1
                ? formatPercent(numericConfidence)
                : `${numericConfidence.toFixed(2)}%`;

            confidenceElement.textContent = displayConfidence;
          } else {
            confidenceElement.textContent = String(confidence);
          }
        } else {
          confidenceElement.textContent = '—';
        }
      }

      safeText(modelElement, modelName);
      safeText(deviceElement, device);

      safeText(
        analysisMessage,
        `${modelName} checkpoint · ${device}`
      );

      safeText(
        resultStatus,
        'Analysis complete'
      );

      /*
       * Optional fields if the dashboard contains them.
       */

      const probabilitiesElement =
        getElement('#probabilities');

      if (
        probabilitiesElement &&
        data.probabilities
      ) {
        probabilitiesElement.textContent =
          JSON.stringify(data.probabilities);
      }

      /*
       * Scroll to result section when available.
       */

      const resultSection =
        getElement('#resultSection') ||
        getElement('#resultsSection') ||
        getElement('#analysisResult');

      if (resultSection) {
        resultSection.scrollIntoView({
          behavior: 'smooth',
          block: 'center'
        });
      }

      showToast('Prediction completed successfully.');
    } catch (error) {
      console.error('Prediction error:', error);

      safeText(
        resultStatus,
        'Analysis failed'
      );

      safeText(
        analysisMessage,
        'The prediction service is currently unavailable.'
      );

      showToast(
        error.message
          ? `${error.message} · backend unavailable`
          : 'Prediction failed · backend unavailable'
      );
    } finally {
      analyzeButton.disabled = false;
    }
  });
}


// ------------------------------------------------------------
// Reset analysis
// ------------------------------------------------------------

const resetButton =
  getElement('#resetButton') ||
  getElement('#resetAnalysis');

if (resetButton) {
  resetButton.addEventListener('click', () => {
    selectedImage = null;

    if (imageInput) {
      imageInput.value = '';
    }

    if (previewImage) {
      previewImage.src = '';
    }

    if (previewContainer) {
      previewContainer.classList.remove('active');
    }

    safeText(
      getElement('#resultStatus'),
      'Ready for analysis'
    );

    safeText(
      getElement('#analysisMessage'),
      'Upload a microscopic lung image to begin.'
    );

    safeText(
      getElement('#prediction') ||
      getElement('#resultPrediction') ||
      getElement('#diagnosis'),
      '—'
    );

    safeText(
      getElement('#confidence') ||
      getElement('#resultConfidence'),
      '—'
    );

    showToast('Analysis reset.');
  });
}


// ------------------------------------------------------------
// Backend health check
// ------------------------------------------------------------

async function checkBackend() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/metrics`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json'
      }
    });

    if (response.ok) {
      console.log(
        'Backend connection successful:',
        API_BASE_URL
      );

      return true;
    }

    console.warn(
      'Backend responded with status:',
      response.status
    );

    return false;
  } catch (error) {
    console.warn(
      'Backend health check failed:',
      error
    );

    return false;
  }
}


// ------------------------------------------------------------
// Initial page setup
// ------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  loadLiveMetrics();
  checkBackend();

  console.log(
    'Microscopic Lung Cancer Detection Dashboard initialized.'
  );

  console.log(
    'API:',
    API_BASE_URL
  );
});