const app = window.Telegram.WebApp;
app.ready();
app.expand();
app.setHeaderColor?.('secondary_bg_color');

const initData = app.initData;
const panel = document.getElementById('panel');
const dialog = document.getElementById('model-dialog');
const search = document.getElementById('model-search');
const results = document.getElementById('model-results');
const count = document.getElementById('model-count');
const detail = document.getElementById('model-detail');
const pickerSubtitle = document.getElementById('picker-subtitle');
let catalogPromise;
let catalog = [];
let capability = '';
let chosen = null;
let selectedRouting = 'auto';
let selectedProvider = '';
let originalModel = '';
const providerCache = new Map();
const capabilityNames = {
  chat: 'General chat', image_understanding: 'Image understanding', video_understanding: 'Video understanding',
  image_generation: 'Image generation', audio_generation: 'Speech generation', transcription: 'Transcription',
  video_generation: 'Video generation'
};

document.addEventListener('htmx:configRequest', event => {
  event.detail.headers['X-Telegram-Init-Data'] = initData;
});

const text = (tag, value, className) => {
  const element = document.createElement(tag);
  if (className) element.className = className;
  element.textContent = value;
  return element;
};

const supports = (model, cap) => {
  const input = model.input_modalities || [];
  const output = model.output_modalities || [];
  if (cap === 'chat') return input.includes('text') && output.includes('text');
  if (cap === 'image_understanding') return input.includes('image') && output.includes('text');
  if (cap === 'video_understanding') return input.includes('video') && output.includes('text');
  if (cap === 'image_generation') return output.includes('image');
  if (cap === 'audio_generation') return output.includes('speech');
  if (cap === 'transcription') return output.includes('transcription');
  if (cap === 'video_generation') return output.includes('video');
  return false;
};

const fuzzyScore = (model, rawQuery) => {
  const query = rawQuery.trim().toLowerCase();
  if (!query) return 1;
  const haystack = `${model.name} ${model.id} ${model.description || ''} ${(model.input_modalities || []).join(' ')} ${(model.output_modalities || []).join(' ')}`.toLowerCase();
  const direct = haystack.indexOf(query);
  if (direct >= 0) return 10000 - direct;
  const tokens = query.split(/\s+/).filter(Boolean);
  if (tokens.every(token => haystack.includes(token))) return 7000 - tokens.reduce((sum, token) => sum + haystack.indexOf(token), 0);
  let at = 0;
  let gaps = 0;
  for (const character of query.replace(/\s/g, '')) {
    const next = haystack.indexOf(character, at);
    if (next < 0) return 0;
    gaps += next - at;
    at = next + 1;
  }
  return Math.max(1, 4000 - gaps);
};

const compactNumber = value => value ? new Intl.NumberFormat(undefined, { notation: 'compact' }).format(value) : '—';
const price = value => {
  if (value === undefined) return '—';
  const numeric = Number(value) * 1_000_000;
  return Number.isFinite(numeric) ? `$${numeric.toLocaleString(undefined, { maximumFractionDigits: 4 })} / 1M` : value;
};
const date = value => value ? new Date(value * 1000).toLocaleDateString() : 'Not published';
const unitPrice = (key, value) => {
  if (['prompt', 'completion', 'input_cache_read', 'input_cache_write', 'audio', 'internal_reasoning'].includes(key)) {
    return `${key}: ${price(value)}`;
  }
  const numeric = Number(value);
  return `${key}: ${Number.isFinite(numeric) ? `$${numeric.toLocaleString(undefined, { maximumFractionDigits: 6 })}` : value}`;
};

const chips = values => {
  const box = text('div', '', 'chips');
  values.filter(Boolean).forEach(value => box.append(text('span', value, 'chip')));
  return box;
};

const fact = (label, value) => {
  const box = text('div', '', 'fact');
  box.append(text('span', label), text('strong', value));
  return box;
};

const providerLabel = endpoint => {
  const parts = [endpoint.name, endpoint.tag];
  if (endpoint.context_length) parts.push(`${compactNumber(endpoint.context_length)} ctx`);
  if (endpoint.uptime !== null && endpoint.uptime !== undefined) parts.push(`${Number(endpoint.uptime).toFixed(1)}% uptime`);
  return parts.filter(Boolean).join(' · ');
};

const loadProviders = async (modelId, select, help, selected, saveButton) => {
  select.disabled = true;
  saveButton.disabled = true;
  try {
    if (!providerCache.has(modelId)) {
      providerCache.set(modelId, fetch(`model-providers?model=${encodeURIComponent(modelId)}`, {
        headers: { 'X-Telegram-Init-Data': initData }
      }).then(async response => {
        if (!response.ok) throw new Error(await response.text());
        return (await response.json()).providers || [];
      }));
    }
    const endpoints = await providerCache.get(modelId);
    const unique = new Map(endpoints.map(endpoint => [endpoint.tag, endpoint]));
    select.replaceChildren(new Option('Auto · any compatible provider', ''));
    unique.forEach(endpoint => select.add(new Option(providerLabel(endpoint), endpoint.tag)));
    if (selected && !unique.has(selected)) select.add(new Option(`${selected} · saved override`, selected));
    select.value = selected;
    help.textContent = `${unique.size} current provider endpoint${unique.size === 1 ? '' : 's'} · Auto is recommended unless you need to pin one.`;
  } catch (error) {
    if (selected) select.add(new Option(`${selected} · saved override`, selected));
    select.value = selected;
    help.textContent = 'Provider endpoints are temporarily unavailable; automatic routing remains available.';
  } finally {
    select.disabled = false;
    saveButton.disabled = false;
  }
};

const showDetail = model => {
  chosen = model;
  detail.replaceChildren();
  detail.append(text('h2', model.name), text('div', model.id, 'model-id'));
  detail.append(chips([...(model.input_modalities || []).map(item => `in: ${item}`), ...(model.output_modalities || []).map(item => `out: ${item}`)]));
  detail.append(text('p', model.description || 'No description is supplied by OpenRouter.', 'description'));
  const facts = text('div', '', 'facts');
  facts.append(
    fact('Context', compactNumber(model.context_length)),
    fact('Max output', compactNumber(model.max_completion_tokens)),
    fact('Input price', price(model.pricing?.prompt)),
    fact('Output price', price(model.pricing?.completion)),
    fact('Knowledge cutoff', model.knowledge_cutoff || 'Not published'),
    fact('Tokenizer', model.tokenizer || 'Not published'),
    fact('Released', date(model.created)),
    fact('Expires', model.expiration_date || 'Not published')
  );
  detail.append(facts);
  const prices = Object.entries(model.pricing || {}).map(([key, value]) => unitPrice(key, value));
  if (prices.length) {
    detail.append(text('h3', 'Published pricing'));
    detail.append(chips(prices));
  }
  const extras = [
    ...(model.supported_resolutions || []).map(item => `resolution: ${item}`),
    ...(model.supported_aspect_ratios || []).map(item => `aspect: ${item}`),
    ...(model.supported_durations || []).map(item => `duration: ${item}`),
    ...(model.supported_sizes || []).map(item => `size: ${item}`),
    ...(model.supported_frame_images || []).map(item => `frame: ${item}`),
    ...(model.generates_audio ? ['generates audio'] : []),
    ...(model.supported_voices || []).map(item => `voice: ${item}`)
  ];
  if (extras.length) detail.append(chips(extras));
  if ((model.supported_parameters || []).length) {
    detail.append(text('h3', 'Supported parameters'));
    detail.append(chips(model.supported_parameters));
  }
  detail.append(document.getElementById('model-settings').content.cloneNode(true));
  const routing = detail.querySelector('[name=routing]');
  routing.value = selectedRouting;
  if (!['chat', 'image_understanding', 'video_understanding'].includes(capability)) {
    routing.querySelector('[value=exacto]')?.remove();
  }
  const saveButton = detail.querySelector('[data-save-model]');
  loadProviders(
    model.id,
    detail.querySelector('[name=provider]'),
    detail.querySelector('.provider-help'),
    model.id === originalModel ? selectedProvider : '',
    saveButton
  );
  saveButton.addEventListener('click', saveModel);
  [...results.children].forEach(button => button.classList.toggle('selected', button.dataset.id === model.id));
};

const renderResults = () => {
  const query = search.value;
  const filtered = catalog
    .filter(model => supports(model, capability))
    .map(model => ({ model, score: fuzzyScore(model, query) }))
    .filter(item => item.score > 0)
    .sort((a, b) => b.score - a.score || (b.model.created || 0) - (a.model.created || 0) || a.model.name.localeCompare(b.model.name));
  count.textContent = `${filtered.length.toLocaleString()} matching models · showing ${Math.min(60, filtered.length)}`;
  results.replaceChildren();
  filtered.slice(0, 60).forEach(({ model }) => {
    const button = text('button', '', 'model-result');
    button.type = 'button';
    button.dataset.id = model.id;
    button.append(text('strong', model.name), text('span', model.id));
    button.addEventListener('click', () => showDetail(model));
    results.append(button);
  });
  if (!filtered.length) results.append(text('div', 'No models match this search.', 'muted'));
};

const loadCatalog = () => {
  catalogPromise ||= fetch('models', { headers: { 'X-Telegram-Init-Data': initData } })
    .then(async response => {
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    })
    .then(body => { catalog = body.models || []; enrichCards(); return catalog; });
  return catalogPromise;
};

const enrichCards = () => {
  document.querySelectorAll('.model-card').forEach(card => {
    const model = catalog.find(item => item.id === card.dataset.model);
    if (!model) return;
    card.querySelector('.model-name').textContent = model.name;
    const summary = card.querySelector('.model-summary');
    summary.textContent = model.description ? `${model.description.slice(0, 155)}${model.description.length > 155 ? '…' : ''}` : 'OpenRouter model metadata is not available.';
    const target = card.querySelector('.model-chips');
    target.replaceChildren(...[
      `context ${compactNumber(model.context_length)}`,
      ...(model.input_modalities || []).map(item => `in: ${item}`),
      ...(model.output_modalities || []).map(item => `out: ${item}`)
    ].map(value => text('span', value, 'chip')));
  });
};

const openPicker = async button => {
  capability = button.dataset.capability;
  pickerSubtitle.textContent = `${capabilityNames[capability] || capability} · live OpenRouter catalog`;
  selectedRouting = button.dataset.routing || 'auto';
  selectedProvider = button.dataset.provider || '';
  originalModel = button.dataset.model;
  search.value = '';
  detail.replaceChildren(text('div', 'Loading model details…', 'loading'));
  dialog.showModal();
  try {
    await loadCatalog();
    renderResults();
    const selected = catalog.find(model => model.id === button.dataset.model && supports(model, capability));
    if (selected) showDetail(selected);
    else detail.replaceChildren(text('div', 'The saved model is no longer in the current OpenRouter catalog. Choose a replacement.', 'catalog-error'));
    search.focus();
  } catch (error) {
    detail.replaceChildren(text('div', `Could not load OpenRouter models: ${error.message}`, 'catalog-error'));
  }
};

const saveModel = async event => {
  const button = event.currentTarget;
  if (!chosen) return;
  button.disabled = true;
  button.textContent = 'Saving…';
  const settings = detail;
  const body = new URLSearchParams({
    capability,
    model: chosen.id,
    routing: settings.querySelector('[name=routing]').value,
    provider: settings.querySelector('[name=provider]').value
  });
  try {
    const response = await fetch('model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded', 'X-Telegram-Init-Data': initData },
      body
    });
    const html = await response.text();
    if (!response.ok) throw new Error(html.replace(/<[^>]+>/g, ''));
    panel.innerHTML = html;
    window.htmx.process(panel);
    enrichCards();
    dialog.close();
    app.HapticFeedback?.notificationOccurred('success');
  } catch (error) {
    button.disabled = false;
    button.textContent = 'Use this model';
    app.showAlert?.(error.message);
  }
};

document.addEventListener('click', event => {
  const picker = event.target.closest('.model-picker');
  if (picker) openPicker(picker);
  const jump = event.target.closest('[data-jump]');
  if (jump) document.getElementById(jump.dataset.jump)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
});
document.getElementById('model-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
search.addEventListener('input', renderResults);
document.addEventListener('htmx:afterSwap', enrichCards);

const adminPath = location.pathname.replace(/\/+$/, '');
window.htmx.ajax('GET', `${adminPath}/panel`, { target: '#panel', swap: 'innerHTML transition:true' });
loadCatalog().catch(() => {});
