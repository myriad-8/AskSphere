document.addEventListener('DOMContentLoaded', () => {
    const qaForm = document.getElementById('qa-form');
    const modelTypeSelect = document.getElementById('model-type');
    const extSettings = document.getElementById('extractive-settings');
    const genSettings = document.getElementById('generative-settings');
    const accordionHeader = document.getElementById('advanced-settings-toggle');
    const resultContainer = document.getElementById('result-container');
    const loader = document.getElementById('loader');
    const resultContent = document.getElementById('result-content');
    const answerText = document.getElementById('answer-text');
    const statsContainer = document.getElementById('stats-container');
    const submitBtn = document.getElementById('submit-btn');

    const banner = document.getElementById('status-banner');
    const exampleBtn = document.getElementById('example-btn');
    let abortController = null;

    const fmt = (v) => (typeof v === 'number' ? v.toFixed(4) : '–');

    const EXAMPLE = {
        context: 'Albert Einstein was a German-born theoretical physicist who developed the theory of relativity. ' +
            'He received the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect.',
        question: 'When did Einstein receive the Nobel Prize?',
    };
    exampleBtn.addEventListener('click', () => {
        document.getElementById('context').value = EXAMPLE.context;
        document.getElementById('question').value = EXAMPLE.question;
    });

    // Ask the server which models are loaded; disable the ones that are not.
    fetch('/api/health').then(r => r.json()).then(h => {
        const missing = [];
        for (const opt of modelTypeSelect.options) {
            const m = h.models[opt.value];
            if (m && !m.loaded) {
                opt.disabled = true;
                opt.textContent += ' (not loaded)';
                missing.push(opt.value);
            }
        }
        if (modelTypeSelect.selectedOptions[0].disabled) {
            const first = [...modelTypeSelect.options].find(o => !o.disabled);
            if (first) { modelTypeSelect.value = first.value; modelTypeSelect.dispatchEvent(new Event('change')); }
        }
        if (missing.length) {
            banner.textContent = missing.length === modelTypeSelect.options.length
                ? 'No model checkpoints found. See the README (“Models”) to add or train them.'
                : `Not loaded: ${missing.join(', ')}. See the README (“Models”).`;
            banner.classList.remove('hidden');
        }
    }).catch(() => {});

    // Toggle Advanced Settings Accordion
    accordionHeader.addEventListener('click', () => {
        accordionHeader.classList.toggle('active');
    });

    // Update range input displays
    const ranges = document.querySelectorAll('input[type="range"]');
    ranges.forEach(range => {
        const display = document.getElementById(`${range.id}-val`);
        range.addEventListener('input', () => {
            display.textContent = range.value;
        });
    });

    // Toggle settings based on model type
    modelTypeSelect.addEventListener('change', (e) => {
        if (e.target.value !== 'generative') {
            extSettings.classList.remove('hidden');
            genSettings.classList.add('hidden');
        } else {
            extSettings.classList.add('hidden');
            genSettings.classList.remove('hidden');
        }
    });

    // Handle enable no-answer gate checkbox
    const enableGateCheckbox = document.getElementById('enable-no-answer-gate');
    const gateThresholdContainer = document.getElementById('gate-threshold-container');
    enableGateCheckbox.addEventListener('change', (e) => {
        if (e.target.checked) {
            gateThresholdContainer.classList.remove('hidden');
        } else {
            gateThresholdContainer.classList.add('hidden');
        }
    });

    // Submit Handler
    qaForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (abortController) {
            abortController.abort(); // Cancel previous request
        }
        abortController = new AbortController();

        const modelType = modelTypeSelect.value;
        const payload = {
            model_type: modelType,
            context: document.getElementById('context').value,
            question: document.getElementById('question').value,
            max_length: parseInt(document.getElementById('max-length').value),
        };

        if (modelType !== 'generative') {
            payload.doc_stride = parseInt(document.getElementById('doc-stride').value);
            payload.n_best = parseInt(document.getElementById('n-best').value);
            payload.max_answer_length = parseInt(document.getElementById('max-answer-length').value);
            payload.no_answer_threshold = parseFloat(document.getElementById('ext-no-answer-threshold').value);
        } else {
            payload.beam_size = parseInt(document.getElementById('beam-size').value);
            payload.max_new_tokens = parseInt(document.getElementById('max-new-tokens').value);
            payload.length_penalty = parseFloat(document.getElementById('length-penalty').value);
            payload.enable_no_answer_gate = enableGateCheckbox.checked;
            payload.no_answer_threshold = parseFloat(document.getElementById('gen-no-answer-threshold').value);
        }

        // UI Reset for loading
        resultContainer.classList.remove('hidden');
        resultContent.classList.add('hidden');
        loader.classList.remove('hidden');
        submitBtn.disabled = true;
        submitBtn.textContent = "Processing...";

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                signal: abortController.signal
            });

            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                throw new Error(typeof err.detail === "string" ? err.detail : "Please check your inputs.");
            }

            const data = await response.json();
            
            // Display Results
            loader.classList.add('hidden');
            resultContent.classList.remove('hidden');
            
            const noAnswer = data.predicted_no_answer || (data.gate && data.gate.selected_no_answer);
            answerText.textContent = data.answer || (noAnswer ? "No answer found in this passage." : "The model returned an empty answer.");
            
            // Build stats
            let statsHtml = '';
            if (modelType !== 'generative') {
                statsHtml += `<span>Span Score: ${fmt(data.span_score)}</span>`;
                statsHtml += `<span>Null Score: ${fmt(data.null_score)}</span>`;
                statsHtml += `<span>Score Diff (Null - Span): ${fmt(data.score_diff_null_minus_span)}</span>`;
                statsHtml += `<span>Predicted No-Answer: ${data.predicted_no_answer ? "Yes" : "No"}</span>`;
            } else {
                if (data.gate && data.gate.enabled !== false) {
                    statsHtml += `<span>Score Diff: ${fmt(data.gate.score_diff)}</span>`;
                    statsHtml += `<span>Predicted No-Answer: ${data.gate.selected_no_answer ? "Yes" : "No"}</span>`;
                }
            }
            statsContainer.innerHTML = statsHtml;

        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('Request cancelled by new submission');
                return; // Don't reset UI, a new request is taking over
            }
            loader.classList.add('hidden');
            resultContent.classList.remove('hidden');
            answerText.textContent = `Error: ${error.message}`;
            statsContainer.innerHTML = '';
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = "Get Answer";
        }
    });
});
