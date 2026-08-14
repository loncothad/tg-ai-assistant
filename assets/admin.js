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
const providerTabs = document.getElementById('model-provider-tabs');
let catalogPromise;
let catalog = [];
let catalogProviders = [];
let activeModelProvider = 'openrouter';
let capability = '';
let chosen = null;
let selectedRouting = 'auto';
let selectedProvider = '';
let originalModel = '';
let originalModelProvider = 'openrouter';
const providerCache = new Map();
const capabilityNames = {
  chat: 'General chat', model_upgrade: 'Advanced model', image_understanding: 'Image understanding', video_understanding: 'Video understanding',
  intent_planning: 'Intent processing', intent_planning_fallback: 'Intent processing fallback',
  output_processing: 'Text output processing', error_processing: 'Error explanation',
  image_generation: 'Image generation', audio_generation: 'Speech generation', speech_generation: 'Speech generation', music_generation: 'Music generation', transcription: 'Transcription',
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
  if (['chat', 'model_upgrade', 'output_processing', 'error_processing'].includes(cap)) {
    return input.includes('text') && output.includes('text');
  }
  if (cap === 'intent_planning' || cap === 'intent_planning_fallback') {
    const parameters = model.supported_parameters || [];
    return input.includes('text') && output.includes('text') &&
      (parameters.includes('response_format') || parameters.includes('structured_outputs'));
  }
  if (cap === 'image_understanding') return input.includes('image') && output.includes('text');
  if (cap === 'video_understanding') return input.includes('video') && output.includes('text');
  if (cap === 'image_generation') return output.includes('image');
  if (cap === 'audio_generation' || cap === 'speech_generation') return output.includes('speech');
  if (cap === 'music_generation') return output.includes('audio');
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
  if (value === undefined || value === null || value === '') return 'Not published';
  const numeric = Number(value) * 1_000_000;
  return Number.isFinite(numeric) ? `$${numeric.toLocaleString(undefined, { maximumFractionDigits: 4 })} / 1M` : value;
};
const date = value => value ? new Date(value * 1000).toLocaleDateString() : 'Not published';
const unitPrice = (key, value) => {
  const mediaEndpoint = ['image_generation', 'audio_generation', 'speech_generation', 'music_generation', 'transcription', 'video_generation'].includes(capability);
  if (['prompt', 'completion'].includes(key) && mediaEndpoint) {
    const numeric = Number(value);
    if (numeric === 0) return `${key}: not billed on this field`;
    return `${key}: ${Number.isFinite(numeric) ? `$${numeric.toLocaleString(undefined, { maximumFractionDigits: 6 })} / published billing unit` : value}`;
  }
  if (['prompt', 'completion', 'input_cache_read', 'input_cache_write', 'internal_reasoning', 'audio', 'audio_output'].includes(key)) {
    return `${key}: ${price(value)}`;
  }
  const numeric = Number(value);
  return `${key}: ${Number.isFinite(numeric) ? `$${numeric.toLocaleString(undefined, { maximumFractionDigits: 6 })} / unit` : value}`;
};

const meaningfulTokenPrice = value => {
  if (value === undefined || value === null || value === '') return 'Not published';
  return Number(value) === 0 ? 'Not token-priced' : price(value);
};

const primaryRate = value => {
  if (value === undefined || value === null || value === '') return 'Not published';
  if (Number(value) === 0) return 'Not billed on this field';
  if (['image_generation', 'audio_generation', 'speech_generation', 'music_generation', 'transcription', 'video_generation'].includes(capability)) {
    return `$${Number(value).toLocaleString(undefined, { maximumFractionDigits: 6 })} / published unit`;
  }
  return meaningfulTokenPrice(value);
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

const loadProviders = async (model, select, help, selected, saveButton) => {
  if (model.model_provider !== 'openrouter') {
    select.replaceChildren(new Option('Direct · AI Hub', ''));
    select.disabled = true;
    help.textContent = 'No secondary endpoint routing is exposed by AI Hub.';
    saveButton.disabled = false;
    return;
  }
  const modelId = model.id;
  select.disabled = true;
  saveButton.disabled = true;
  try {
    const cacheKey = `${capability}:${modelId}`;
    if (!providerCache.has(cacheKey)) {
      providerCache.set(cacheKey, fetch(`model-providers?model=${encodeURIComponent(modelId)}&model_provider=openrouter&capability=${encodeURIComponent(capability)}`, {
        headers: { 'X-Telegram-Init-Data': initData }
      }).then(async response => {
        if (!response.ok) throw new Error(await response.text());
        return (await response.json()).providers || [];
      }));
    }
    const endpoints = await providerCache.get(cacheKey);
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
  const providerName = model.model_provider === 'aihub' ? 'AI Hub' : 'OpenRouter';
  detail.append(text('h2', model.name), text('div', `${providerName} · ${model.id}`, 'model-id'));
  detail.append(chips([...(model.input_modalities || []).map(item => `in: ${item}`), ...(model.output_modalities || []).map(item => `out: ${item}`)]));
  detail.append(text('p', model.description || `No description is supplied by ${providerName}.`, 'description'));
  const facts = text('div', '', 'facts');
  facts.append(
    fact('Context', compactNumber(model.context_length)),
    fact('Max output', compactNumber(model.max_completion_tokens)),
    fact('Input rate', primaryRate(model.pricing?.prompt)),
    fact('Output rate', primaryRate(model.pricing?.completion)),
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
  const openrouterControls = detail.querySelector('.openrouter-routing');
  const directNote = detail.querySelector('.direct-provider-note');
  const isOpenRouter = model.model_provider === 'openrouter';
  openrouterControls.hidden = !isOpenRouter;
  directNote.hidden = isOpenRouter;
  routing.value = isOpenRouter ? selectedRouting : 'auto';
  if (!['chat', 'image_understanding', 'video_understanding'].includes(capability)) routing.querySelector('[value=exacto]')?.remove();
  const saveButton = detail.querySelector('[data-save-model]');
  loadProviders(
    model,
    detail.querySelector('[name=provider]'),
    detail.querySelector('.provider-help'),
    model.id === originalModel && model.model_provider === originalModelProvider ? selectedProvider : '',
    saveButton
  );
  saveButton.addEventListener('click', saveModel);
  [...results.children].forEach(button => button.classList.toggle('selected', button.dataset.id === model.id));
};

const renderResults = () => {
  const query = search.value;
  const filtered = catalog
    .filter(model => model.model_provider === activeModelProvider)
    .filter(model => supports(model, capability))
    .map(model => ({ model, score: fuzzyScore(model, query) }))
    .filter(item => item.score > 0)
    .sort((a, b) => b.score - a.score || (b.model.created || 0) - (a.model.created || 0) || a.model.name.localeCompare(b.model.name));
  const provider = catalogProviders.find(item => item.id === activeModelProvider);
  count.textContent = `${provider?.label || activeModelProvider} · ${filtered.length.toLocaleString()} matching models · showing ${Math.min(60, filtered.length)}`;
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

const renderProviderTabs = () => {
  providerTabs.replaceChildren();
  catalogProviders.forEach(provider => {
    const button = text('button', `${provider.label}${provider.available ? ` · ${provider.models || 0}` : ' · unavailable'}`);
    button.type = 'button';
    button.role = 'tab';
    button.disabled = !provider.available;
    button.classList.toggle('selected', provider.id === activeModelProvider);
    button.addEventListener('click', () => {
      activeModelProvider = provider.id;
      chosen = null;
      renderProviderTabs();
      renderResults();
      detail.replaceChildren(text('div', `Choose a ${provider.label} model to inspect it.`, 'muted'));
      pickerSubtitle.textContent = `${capabilityNames[capability] || capability} · ${provider.label}`;
    });
    providerTabs.append(button);
  });
};

const loadCatalog = () => {
  catalogPromise ||= fetch('models', { headers: { 'X-Telegram-Init-Data': initData } })
    .then(async response => {
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    })
    .then(body => {
      catalog = body.models || [];
      catalogProviders = body.providers || [];
      enrichCards();
      return catalog;
    });
  return catalogPromise;
};

const enrichCards = () => {
  document.querySelectorAll('.model-card').forEach(card => {
    const model = catalog.find(item => item.id === card.dataset.model && item.model_provider === card.dataset.modelProvider);
    if (!model) return;
    card.querySelector('.model-name').textContent = model.name;
    const summary = card.querySelector('.model-summary');
    summary.textContent = model.description ? `${model.description.slice(0, 155)}${model.description.length > 155 ? '…' : ''}` : 'Provider metadata is not available.';
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
  activeModelProvider = button.dataset.modelProvider || 'openrouter';
  pickerSubtitle.textContent = `${capabilityNames[capability] || capability} · provider catalogs`;
  selectedRouting = button.dataset.routing || 'auto';
  selectedProvider = button.dataset.provider || '';
  originalModel = button.dataset.model;
  originalModelProvider = button.dataset.modelProvider || 'openrouter';
  search.value = '';
  detail.replaceChildren(text('div', 'Loading model details…', 'loading'));
  dialog.showModal();
  try {
    await loadCatalog();
    if (!catalogProviders.some(provider => provider.id === activeModelProvider && provider.available)) {
      activeModelProvider = catalogProviders.find(provider => provider.available)?.id || activeModelProvider;
    }
    renderProviderTabs();
    renderResults();
    const selected = catalog.find(model => model.id === button.dataset.model && model.model_provider === button.dataset.modelProvider && supports(model, capability));
    if (selected) showDetail(selected);
    else detail.replaceChildren(text('div', 'The saved model is no longer in its current provider catalog. Choose a replacement.', 'catalog-error'));
    search.focus();
  } catch (error) {
    detail.replaceChildren(text('div', `Could not load model catalogs: ${error.message}`, 'catalog-error'));
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
    model_provider: chosen.model_provider,
    routing: chosen.model_provider === 'openrouter' ? settings.querySelector('[name=routing]').value : 'auto',
    provider: chosen.model_provider === 'openrouter' ? settings.querySelector('[name=provider]').value : ''
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
document.addEventListener('htmx:afterSwap', event => {
  if (event.detail.target !== panel) return;
  catalogPromise = null;
  providerCache.clear();
  loadCatalog().catch(() => {});
});

const adminPath = location.pathname.replace(/\/+$/, '');
window.htmx.ajax('GET', `${adminPath}/panel`, { target: '#panel', swap: 'innerHTML transition:true' });
loadCatalog().catch(() => {});
