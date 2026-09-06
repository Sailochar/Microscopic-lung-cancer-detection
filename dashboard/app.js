const API_BASE_URL = 'https://microscopic-lung-cancer-detection-1.onrender.com';

const authBackdrop = document.querySelector('#authBackdrop');
const authForm = document.querySelector('#authForm');
const nameField = document.querySelector('#nameField');
const authTitle = document.querySelector('#authTitle');
const authSubmitText = document.querySelector('#authSubmitText');
const toast = document.querySelector('#toast');
const userName = document.querySelector('#userName');
const uploadZone = document.querySelector('#uploadZone');
const imageInput = document.querySelector('#imageInput');
const uploadPreview = document.querySelector('#uploadPreview');

let authMode = 'signin';

function showToast(message) {
  toast.textContent = message;
  toast.classList.add('visible');
  window.setTimeout(() => toast.classList.remove('visible'), 2800);
}

const reportPrompt = document.querySelector('#reportPrompt');
const viewReportButton = document.querySelector('#viewReportButton');
const stayScreeningButton = document.querySelector('#stayScreeningButton');

function setImage(source, label) {
  uploadPreview.innerHTML = `<img src="${source}" alt="${label}">`;
  document.querySelector('#uploadTitle').textContent = label;
  document.querySelector('#uploadHint').textContent =
    'Ready for review · image stays in this browser session';
}

function enterWorkspace(name) {
  document.body.classList.add('authenticated');
  userName.textContent = name || 'Dr. Rao';
  localStorage.setItem('privcanf_user', name || 'Dr. Rao');
}

document.querySelectorAll('.auth-tab').forEach((tab) =>
  tab.addEventListener('click', () => {
    authMode = tab.dataset.mode;

    document
      .querySelectorAll('.auth-tab')
      .forEach((item) =>
        item.classList.toggle('active', item === tab)
      );

    const isSignup = authMode === 'signup';

    nameField.classList.toggle('hidden', !isSignup);
    document.querySelector('#nameInput').required = isSignup;

    authTitle.textContent = isSignup
      ? 'Create your private workspace.'
      : 'A clearer view of every case.';

    authSubmitText.textContent = isSignup
      ? 'Create account'
      : 'Sign in to workspace';
  })
);

document.querySelector('#demoButton').addEventListener('click', () => {
  enterWorkspace('Demo user');
  showToast('Demo workspace ready');
});

authForm.addEventListener('submit', (event) => {
  event.preventDefault();

  const name =
    authMode === 'signup'
      ? document.querySelector('#nameInput').value
      : 'Dr. Rao';

  enterWorkspace(name);

  showToast(
    authMode === 'signup'
      ? 'Workspace created locally'
      : 'Welcome back'
  );
});

document.querySelector('#logoutButton').addEventListener('click', () => {
  document.body.classList.remove('authenticated');
  localStorage.removeItem('privcanf_user');
});

document.querySelector('#profileButton').addEventListener('click', () =>
  showToast('Signed in as ' + userName.textContent)
);

const navItems = document.querySelectorAll('.nav-item');
const overviewView = document.querySelector('#overviewView');
const screeningView = document.querySelector('#screeningView');
const reportsView = document.querySelector('#reportsView');
const modelHealthView = document.querySelector('#models');
const metricsChart = document.querySelector('#metricsChart');
const chartTooltip = document.querySelector('#chartTooltip');

let metricsLoaded = false;

function switchView(view) {
  const showModelHealth = view === 'models';
  const showScreening = view === 'screening';
  const showReports = view === 'reports';

  overviewView.hidden =
    showModelHealth || showScreening || showReports;

  screeningView.hidden = !showScreening;
  reportsView.hidden = !showReports;
  modelHealthView.hidden = !showModelHealth;

  document
    .querySelector('.page-wrap')
    .classList.toggle(
      'model-health-active',
      showModelHealth
    );

  const activeView = showModelHealth
    ? modelHealthView
    : showScreening
      ? screeningView
      : showReports
        ? reportsView
        : overviewView;

  activeView.classList.remove('view-enter');
  void activeView.offsetWidth;
  activeView.classList.add('view-enter');

  window.scrollTo({
    top: 0,
    behavior: 'smooth'
  });

  if (showModelHealth && !metricsLoaded) {
    loadLiveMetrics();
  }
}

function showReportPrompt(prediction) {
  document.querySelector('#reportPromptText').textContent =
    `${prediction} classification is ready to review. Open the full confidence report now?`;

  reportPrompt.hidden = false;
  viewReportButton.focus();
}

function hideReportPrompt() {
  reportPrompt.hidden = true;
}

viewReportButton.addEventListener('click', () => {
  hideReportPrompt();

  navItems.forEach((item) =>
    item.classList.toggle(
      'active',
      item.getAttribute('href') === '#reports'
    )
  );

  switchView('reports');
});

stayScreeningButton.addEventListener(
  'click',
  hideReportPrompt
);

reportPrompt.addEventListener('click', (event) => {
  if (event.target === reportPrompt) {
    hideReportPrompt();
  }
});

document.addEventListener('keydown', (event) => {
  if (
    event.key === 'Escape' &&
    !reportPrompt.hidden
  ) {
    hideReportPrompt();
  }
});


/* =========================================================
   METRICS
   ========================================================= */

function loadLiveMetrics() {
  const status =
    document.querySelector('#chartLiveStatus');

  fetch(`${API_BASE_URL}/api/metrics`)
    .then(async (response) => {
      let data;

      try {
        data = await response.json();
      } catch {
        throw new Error(
          `Backend returned HTTP ${response.status} instead of JSON`
        );
      }

      return {
        ok: response.ok,
        data
      };
    })
    .then(({ ok, data }) => {
      if (!ok) {
        throw new Error(
          data.error || 'Metrics unavailable'
        );
      }

      if (!Array.isArray(data.metrics)) {
        throw new Error(
          'Invalid metrics response from backend'
        );
      }

      renderMetrics(data.metrics);

      metricsLoaded = true;

      status.textContent =
        `Live · ${data.device}`;
    })
    .catch((error) => {
      document.querySelector('#chartLoading').textContent =
        error.message;

      status.textContent =
        'Metrics unavailable';
    });
}


/* =========================================================
   METRICS CHART
   ========================================================= */

function renderMetrics(metrics) {
  const series = [
    ['accuracy', 'Accuracy', 'accuracy'],
    ['precision', 'Precision', 'precision'],
    ['recall', 'Recall', 'recall'],
    ['macroF1', 'Macro-F1', 'macro-f1']
  ];

  metricsChart.innerHTML =
    `<div class="chart-floor" aria-hidden="true"></div>` +
    metrics
      .map(
        (entry) =>
          `<div class="bar-group">
            <strong>${entry.model}</strong>
            <div class="bar-cluster">
              ${series
                .map(
                  ([key, label, color]) =>
                    `<button
                      class="metric-bar ${color}"
                      style="--bar-height:${Math.max(
                        4,
                        entry[key] * 100
                      )}%"
                      data-model="${entry.model}"
                      data-label="${label}"
                      data-value="${entry[key]}"
                      aria-label="${entry.model} ${label} ${(entry[key] * 100).toFixed(2)} percent"
                    >
                      <span></span>
                    </button>`
                )
                .join('')}
            </div>
          </div>`
      )
      .join('');

  metricsChart
    .querySelectorAll('.metric-bar')
    .forEach((bar) => {
      const showValue = () => {
        chartTooltip.textContent =
          `${bar.dataset.model} · ${bar.dataset.label}: ` +
          `${(Number(bar.dataset.value) * 100).toFixed(2)}%`;

        chartTooltip.classList.add('visible');
        bar.classList.add('focused');
      };

      const hideValue = () => {
        chartTooltip.classList.remove('visible');
        bar.classList.remove('focused');
      };

      bar.addEventListener(
        'pointerenter',
        showValue
      );

      bar.addEventListener(
        'focus',
        showValue
      );

      bar.addEventListener(
        'pointerleave',
        hideValue
      );

      bar.addEventListener(
        'blur',
        hideValue
      );
    });
}


/* =========================================================
   NAVIGATION
   ========================================================= */

navItems.forEach((item) =>
  item.addEventListener('click', (event) => {
    event.preventDefault();

    navItems.forEach((navItem) =>
      navItem.classList.toggle(
        'active',
        navItem === item
      )
    );

    const target =
      item.getAttribute('href').slice(1);

    if (target === 'models') {
      switchView('models');
    } else if (target === 'screening') {
      switchView('screening');
    } else if (target === 'reports') {
      switchView('reports');
    } else {
      switchView('overview');

      const section =
        document.querySelector(`#${target}`);

      if (
        section &&
        target !== 'overview'
      ) {
        window.setTimeout(
          () =>
            section.scrollIntoView({
              behavior: 'smooth',
              block: 'start'
            }),
          80
        );
      }
    }
  })
);


/* =========================================================
   SCREENING LINKS
   ========================================================= */

document
  .querySelectorAll(
    'a.primary-button[href="#screening"]'
  )
  .forEach((link) =>
    link.addEventListener(
      'click',
      (event) => {
        event.preventDefault();

        navItems.forEach((item) =>
          item.classList.toggle(
            'active',
            item.getAttribute('href') ===
              '#screening'
          )
        );

        switchView('screening');
      }
    )
  );


/* =========================================================
   INTERSECTION OBSERVER
   ========================================================= */

const sectionLinks = [...navItems]
  .filter(
    (item) =>
      item.getAttribute('href') !== '#models'
  )
  .map((item) => ({
    item,
    section: document.querySelector(
      item.getAttribute('href')
    )
  }))
  .filter(({ section }) => section);

const sectionObserver =
  new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter(
          (entry) => entry.isIntersecting
        )
        .sort(
          (a, b) =>
            b.intersectionRatio -
            a.intersectionRatio
        )[0];

      if (!visible) return;

      const match =
        sectionLinks.find(
          ({ section }) =>
            section === visible.target
        );

      if (match) {
        navItems.forEach((item) =>
          item.classList.toggle(
            'active',
            item === match.item
          )
        );
      }
    },
    {
      rootMargin:
        '-18% 0px -62% 0px',
      threshold: [0, 0.25, 0.6]
    }
  );

sectionLinks.forEach(
  ({ section }) =>
    sectionObserver.observe(section)
);


/* =========================================================
   THEME
   ========================================================= */

const themeToggle =
  document.querySelector('#themeToggle');

const savedTheme =
  localStorage.getItem(
    'privcanf_theme'
  ) || 'dark';

function applyTheme(theme) {
  const light = theme === 'light';

  document.body.classList.toggle(
    'light-theme',
    light
  );

  themeToggle.setAttribute(
    'aria-pressed',
    String(light)
  );

  themeToggle.setAttribute(
    'aria-label',
    light
      ? 'Switch to dark mode'
      : 'Switch to light mode'
  );

  themeToggle.querySelector(
    '.theme-label'
  ).textContent =
    light ? 'Dark' : 'Light';

  themeToggle.querySelector(
    '.theme-icon'
  ).textContent =
    light ? '☾' : '☼';

  localStorage.setItem(
    'privcanf_theme',
    theme
  );
}

themeToggle.addEventListener(
  'click',
  () => {
    applyTheme(
      document.body.classList.contains(
        'light-theme'
      )
        ? 'dark'
        : 'light'
    );
  }
);

applyTheme(savedTheme);


/* =========================================================
   IMAGE UPLOAD
   ========================================================= */

document
  .querySelector('#browseButton')
  .addEventListener(
    'click',
    () => imageInput.click()
  );

uploadZone.addEventListener(
  'click',
  (event) => {
    if (
      event.target !==
      document.querySelector(
        '#browseButton'
      )
    ) {
      imageInput.click();
    }
  }
);

uploadZone.addEventListener(
  'keydown',
  (event) => {
    if (
      event.key === 'Enter' ||
      event.key === ' '
    ) {
      event.preventDefault();
      imageInput.click();
    }
  }
);

uploadZone.addEventListener(
  'dragover',
  (event) => {
    event.preventDefault();
    uploadZone.classList.add('dragging');
  }
);

uploadZone.addEventListener(
  'dragleave',
  () =>
    uploadZone.classList.remove(
      'dragging'
    )
);

uploadZone.addEventListener(
  'drop',
  (event) => {
    event.preventDefault();

    uploadZone.classList.remove(
      'dragging'
    );

    handleFile(
      event.dataTransfer.files[0]
    );
  }
);

imageInput.addEventListener(
  'change',
  () =>
    handleFile(
      imageInput.files[0]
    )
);

function handleFile(file) {
  if (!file) return;

  if (
    !file.type.startsWith('image/')
  ) {
    showToast(
      'Please choose a JPG, PNG, or WEBP image'
    );
    return;
  }

  if (
    file.size >
    10 * 1024 * 1024
  ) {
    showToast(
      'Image must be smaller than 10 MB'
    );
    return;
  }

  const reader =
    new FileReader();

  reader.addEventListener(
    'load',
    () =>
      setImage(
        reader.result,
        file.name
      )
  );

  reader.readAsDataURL(file);
}


/* =========================================================
   PREDICTION
   ========================================================= */

document
  .querySelector('#analyzeButton')
  .addEventListener(
    'click',
    () => {
      const button =
        document.querySelector(
          '#analyzeButton'
        );

      const image =
        uploadPreview.querySelector(
          'img'
        );

      if (!image) {
        showToast(
          'Upload an image before analyzing'
        );
        return;
      }

      button.disabled = true;

      button.innerHTML =
        '<span aria-hidden="true">⋯</span> Reviewing';

      document.querySelector(
        '#resultStatus'
      ).textContent =
        'Running model';

      const selection =
        document.querySelector(
          '#modelSelect'
        ).value;

      const model =
        selection.startsWith('FedAvg')
          ? 'fedavg'
          : selection.startsWith('Hospital 1')
            ? 'hospital1'
            : selection.startsWith('Hospital 2')
              ? 'hospital2'
              : selection.startsWith('Hospital 3')
                ? 'hospital3'
                : 'fedprox';

      fetch(
        `${API_BASE_URL}/api/predict`,
        {
          method: 'POST',

          headers: {
            'Content-Type':
              'application/json'
          },

          body: JSON.stringify({
            image: image.src,
            model
          })
        }
      )
        .then(async (response) => {
          let data;

          try {
            data =
              await response.json();
          } catch {
            throw new Error(
              `Backend returned HTTP ${response.status} instead of JSON`
            );
          }

          return {
            ok: response.ok,
            data
          };
        })

        .then(({ ok, data }) => {
          if (!ok) {
            throw new Error(
              data.error ||
                'Inference failed'
            );
          }

          if (
            !data.prediction ||
            !data.probabilities
          ) {
            throw new Error(
              'Invalid prediction response from backend'
            );
          }

          updateResult(data);

          button.disabled = false;

          button.innerHTML =
            '<span aria-hidden="true">⌁</span> Analyze image';

          document.querySelector(
            '#resultStatus'
          ).textContent =
            'Analysis complete';

          document.querySelector(
            '#caseId'
          ).textContent =
            'PCF-' +
            String(Date.now()).slice(
              -9
            );

          document.querySelector(
            '#analysisMessage'
          ).textContent =
            `${data.model} checkpoint · ${data.device}`;

          showToast(
            'Report updated from the PyTorch checkpoint'
          );

          showReportPrompt(
            data.prediction
          );
        })

        .catch((error) => {
          button.disabled = false;

          button.innerHTML =
            '<span aria-hidden="true">⌁</span> Analyze image';

          document.querySelector(
            '#resultStatus'
          ).textContent =
            error.message.startsWith(
              'Unsupported image'
            )
              ? 'Image rejected'
              : 'Inference unavailable';

          document.querySelector(
            '#resultClass'
          ).textContent =
            'No report generated';

          document.querySelector(
            '#confidenceValue'
          ).textContent = '—';

          document.querySelector(
            '#resultDescription'
          ).textContent =
            error.message;

          [
            'Aca',
            'N',
            'Scc'
          ].forEach(
            (suffix) => {
              document.querySelector(
                `#prob${suffix}`
              ).style.width =
                '0%';

              document.querySelector(
                `#prob${suffix}Value`
              ).textContent =
                '—';
            }
          );

          showToast(
            error.message
          );
        });
    }
  );


/* =========================================================
   DISPLAY RESULT
   ========================================================= */

function updateResult(data) {
  const percentage =
    (value) =>
      `${(value * 100).toFixed(1)}%`;

  document.querySelector(
    '#resultClass'
  ).textContent =
    data.prediction;

  document.querySelector(
    '#confidenceValue'
  ).textContent =
    percentage(
      data.confidence
    );

  document.querySelector(
    '#resultDescription'
  ).textContent =
    data.prediction === 'lung_n'
      ? 'No malignant pattern detected by this checkpoint.'
      : 'Model detected a malignant pattern. Review with a qualified clinician.';

  [
    ['Aca', 'lung_aca'],
    ['N', 'lung_n'],
    ['Scc', 'lung_scc']
  ].forEach(
    ([suffix, key]) => {
      const value =
        data.probabilities[key];

      document.querySelector(
        `#prob${suffix}`
      ).style.width =
        `${value * 100}%`;

      document.querySelector(
        `#prob${suffix}Value`
      ).textContent =
        percentage(value);
    }
  );
}


/* =========================================================
   DOWNLOAD REPORT
   ========================================================= */

document
  .querySelector('#downloadButton')
  .addEventListener(
    'click',
    () => {
      const report =
        'PrivCanFed screening report\n' +
        'Case: ' +
        document.querySelector(
          '#caseId'
        ).textContent +
        '\nResult: ' +
        document.querySelector(
          '#resultClass'
        ).textContent +
        '\nConfidence: ' +
        document.querySelector(
          '#confidenceValue'
        ).textContent +
        '\nModel: ' +
        document.querySelector(
          '#modelSelect'
        ).value +
        '\n\nResearch use only. Not a standalone diagnosis.';

      const link =
        document.createElement(
          'a'
        );

      link.href =
        URL.createObjectURL(
          new Blob(
            [report],
            {
              type: 'text/plain'
            }
          )
        );

      link.download =
        'privcanf-report.txt';

      link.click();

      URL.revokeObjectURL(
        link.href
      );

      showToast(
        'Report downloaded'
      );
    }
  );


/* =========================================================
   RESTORE USER
   ========================================================= */

const savedUser =
  localStorage.getItem(
    'privcanf_user'
  );

if (savedUser) {
  enterWorkspace(savedUser);
}


/* =========================================================
   TILT EFFECT
   ========================================================= */

const tiltTargets =
  document.querySelectorAll(
    '.panel, .metric-card'
  );

if (
  !window.matchMedia(
    '(prefers-reduced-motion: reduce)'
  ).matches &&
  window.matchMedia(
    '(pointer: fine)'
  ).matches
) {
  tiltTargets.forEach(
    (target) => {
      target.classList.add(
        'tilt-ready'
      );

      target.addEventListener(
        'pointermove',
        (event) => {
          const bounds =
            target.getBoundingClientRect();

          const x =
            (event.clientX -
              bounds.left) /
              bounds.width -
            0.5;

          const y =
            (event.clientY -
              bounds.top) /
              bounds.height -
            0.5;

          target.style.setProperty(
            '--rx',
            `${(-y * 3).toFixed(2)}deg`
          );

          target.style.setProperty(
            '--ry',
            `${(x * 4).toFixed(2)}deg`
          );
        }
      );

      target.addEventListener(
        'pointerleave',
        () => {
          target.style.setProperty(
            '--rx',
            '0deg'
          );

          target.style.setProperty(
            '--ry',
            '0deg'
          );
        }
      );
    }
  );
}