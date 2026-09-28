const $ = selector => document.querySelector(selector);
const selectedSignFiles = [];
let inputObjectUrl = null;
let isProcessing = false;

function escapeHtml(value = '') {
  return String(value).replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
}

function setTrace(open) {
  $('#trace-panel').hidden = !open;
  $('#trace-toggle').setAttribute('aria-expanded', String(open));
  if (open) $('#trace-panel').scrollIntoView({behavior: 'smooth', block: 'start'});
}

$('#trace-toggle').addEventListener('click', () => setTrace($('#trace-panel').hidden));
$('#trace-close').addEventListener('click', () => setTrace(false));

function addMessage(role, text, meta = '') {
  const thread = $('#chat-thread');
  const message = document.createElement('div');
  message.className = `message ${role === 'user' ? 'user-message' : 'assistant-message'}`;
  message.innerHTML = `<span class="message-label">${role === 'user' ? 'المستخدم' : 'وصال'}${meta ? ` · ${escapeHtml(meta)}` : ''}</span><p>${escapeHtml(text)}</p>`;
  thread.appendChild(message);
  thread.scrollTop = thread.scrollHeight;
}

function resetOutputVideo() {
  const video = $('#conversation-video');
  video.pause();
  video.removeAttribute('src');
  video.load();
  $('#output-stage').classList.add('empty-stage');
  $('#output-placeholder').hidden = false;
  $('#output-video-state').textContent = 'ستظهر هنا';
}

function clearSelection() {
  selectedSignFiles.length = 0;
  $('#sign-files').value = '';
  if (inputObjectUrl) URL.revokeObjectURL(inputObjectUrl);
  inputObjectUrl = null;
  const video = $('#input-video');
  video.pause(); video.removeAttribute('src'); video.load();
  $('#input-stage').classList.add('empty-stage');
  $('#upload-placeholder').hidden = false;
  $('#input-video-state').textContent = 'بانتظار فيديو';
  $('#selected-sign-files').textContent = 'لم يتم اختيار فيديو بعد';
  $('#run-conversation').disabled = true;
}

function renderSelection() {
  const continuous = $('#continuous-video');
  if (!selectedSignFiles.length) return clearSelection();
  if (selectedSignFiles.length > 1) continuous.checked = false;
  $('#selected-sign-files').textContent = selectedSignFiles.map((file, i) => `${i + 1}. ${file.name}`).join(' · ');
  $('#run-conversation').disabled = false;
  $('#input-video-state').textContent = selectedSignFiles.length === 1 ? 'فيديو جاهز' : `${selectedSignFiles.length} مقاطع جاهزة`;
  if (inputObjectUrl) URL.revokeObjectURL(inputObjectUrl);
  inputObjectUrl = URL.createObjectURL(selectedSignFiles[0]);
  const video = $('#input-video');
  video.src = inputObjectUrl;
  video.muted = true;
  video.autoplay = true;
  $('#input-stage').classList.remove('empty-stage');
  $('#upload-placeholder').hidden = true;
  video.onloadeddata = () => {
    $('#input-video-state').textContent = 'يُعرض الآن';
    $('#conversation-status').textContent = 'سيبدأ التحليل تلقائياً عند انتهاء الفيديو.';
    video.play().catch(() => {
      $('#input-video-state').textContent = 'اضغط تشغيل';
      $('#conversation-status').textContent = 'اضغط تشغيل على الفيديو؛ سيبدأ التحليل تلقائياً عند انتهائه.';
    });
  };
  video.onended = () => {
    $('#input-video-state').textContent = 'انتهى العرض';
    runConversation();
  };
  video.load();
}

function addSelectedFiles(files) {
  for (const file of files) {
    if (!file.type.startsWith('video/')) continue;
    if (!selectedSignFiles.some(x => x.name === file.name && x.size === file.size && x.lastModified === file.lastModified)) {
      selectedSignFiles.push(file);
    }
  }
  renderSelection();
}

$('#sign-files').addEventListener('change', event => {
  addSelectedFiles(event.target.files);
  event.target.value = '';
});
const inputStage = $('#input-stage');
for (const eventName of ['dragenter', 'dragover']) {
  inputStage.addEventListener(eventName, event => { event.preventDefault(); inputStage.classList.add('dragging'); });
}
for (const eventName of ['dragleave', 'drop']) {
  inputStage.addEventListener(eventName, event => { event.preventDefault(); inputStage.classList.remove('dragging'); });
}
inputStage.addEventListener('drop', event => addSelectedFiles(event.dataTransfer.files));
$('#clear-sign-files').addEventListener('click', clearSelection);
$('#clear-chat').addEventListener('click', () => {
  $('#chat-thread').innerHTML = '<div class="message assistant-message"><span class="message-label">وصال</span><p>مرحباً، ارفع فيديو لسؤالك بلغة الإشارة وسأجيبك بالعربية والإشارة.</p></div>';
  $('#conversation-result').innerHTML = '<div class="trace-empty">شغّل محادثة لعرض التعرف وRAG وتحويل الإشارات ومطابقة القاموس هنا.</div>';
  resetOutputVideo(); clearSelection();
});

async function loadRecognitionVocabulary() {
  try {
    const response = await fetch('/api/recognition-vocabulary');
    const words = await response.json();
    if (!response.ok) throw new Error();
    $('#recognition-vocabulary').textContent = words.map(item => item.arabic).join(' · ');
  } catch {
    $('#recognition-vocabulary').textContent = 'تعذر تحميل قائمة الكلمات. تأكد من تشغيل الخادم.';
  }
}
loadRecognitionVocabulary();

function renderTrace(data, predictions, continuous) {
  const trace = $('#conversation-result');
  let html = '';
  if (continuous) {
    const boundaries = data.segmentation || [];
    html += `<div class="stage"><b>التقسيم الزمني</b><p>${boundaries.length} مقطع/كلمة</p>${boundaries.map((x,i) => `<p>${i+1}. ${x.start_seconds.toFixed(2)}s – ${x.end_seconds.toFixed(2)}s</p>`).join('')}</div>`;
  }
  html += `<div class="stage"><b>1. التعرّف على الإشارات</b><p>${predictions.map(x => `${escapeHtml(x.gloss || 'غير معروف')} <span class="score">${(x.confidence*100).toFixed(1)}%</span>`).join(' ← ')}</p></div>`;
  if (data.status === 'review_required') {
    html += `<div class="stage warning"><b>تحتاج إلى مراجعة</b><p>${escapeHtml(data.message)}</p></div>`;
    trace.innerHTML = html; return;
  }
  html += `<div class="stage"><b>2. السؤال العربي</b><p>${escapeHtml(data.reconstructed_question)}</p></div>`;
  html += `<div class="stage"><b>3. سياق RAG</b>${(data.retrieved_context || []).map(x => `<p>${escapeHtml(x.text)}</p>`).join('')}</div>`;
  html += `<div class="stage"><b>4. إجابة RAG</b><p>${escapeHtml(data.answer)}</p></div>`;
  html += `<div class="stage"><b>5. جملة الإشارة السعودية من LLM</b><p>${escapeHtml(data.sign_language_sentence || data.simplified_sentence || '')}</p><p>${(data.answer_glosses || []).map(escapeHtml).join(' ← ')}</p></div>`;
  const matches = (data.word_matches || data.matches || []).map(x => {
    if (x.fallback === 'fingerspell') return `<p><b>${escapeHtml(x.query)}</b> ← تهجئة: ${[...x.query].filter(c => !/\s/.test(c)).map(escapeHtml).join(' ← ')}</p>`;
    if (x.fallback === 'unmapped_character') return `<p>${escapeHtml(x.query)} ← حرف غير متاح</p>`;
    if (x.score < .999) return `<p>${escapeHtml(x.query)} ← ${escapeHtml(x.gloss)} <span class="score">${(x.score*100).toFixed(1)}%</span></p>`;
    return `<p>${escapeHtml(x.query)} ← ${escapeHtml(x.gloss || x.query)}</p>`;
  }).join('');
  html += `<div class="stage"><b>6. القاموس والتهجئة</b>${matches}</div>`;
  html += '<div class="stage"><b>7. بناء فيديو الإجابة</b><p>تم دمج مقاطع القاموس والتهجئة في فيديو واحد متوافق مع المتصفح.</p></div>';
  trace.innerHTML = html;
}

async function runConversation() {
  if (!selectedSignFiles.length || isProcessing) return;
  isProcessing = true;
  const button = $('#run-conversation');
  const status = $('#conversation-status');
  const continuous = $('#continuous-video').checked;
  const form = new FormData();
  selectedSignFiles.forEach(file => form.append('files', file));
  button.disabled = true;
  button.querySelector('span').textContent = 'جارٍ الفهم…';
  status.textContent = 'يتعرّف وصال على الإشارات ويبحث عن الإجابة…';
  $('#input-video-state').textContent = 'قيد التحليل';
  $('#output-video-state').textContent = 'جارٍ الإنشاء';
  try {
    const response = await fetch(`/api/sign-conversation?allow_low_confidence=${$('#allow-low').checked}&continuous_video=${continuous}`, {method:'POST', body:form});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'تعذر إكمال المحادثة');
    const predictions = data.recognition || data.predictions || [];
    renderTrace(data, predictions, continuous);
    if (data.status === 'review_required') {
      status.textContent = 'درجة الثقة منخفضة. افتح تتبّع المعالجة للمراجعة.';
      setTrace(true); return;
    }
    addMessage('user', data.reconstructed_question, predictions.map(x => x.gloss).filter(Boolean).join(' · '));
    addMessage('assistant', data.answer, data.sign_language_sentence || 'إجابة إشارية');
    if (data.video_path) {
      const file = data.video_path.split(/[\\/]/).pop();
      const video = $('#conversation-video');
      $('#output-stage').classList.remove('empty-stage');
      $('#output-placeholder').hidden = true;
      video.hidden = false;
      video.style.visibility = 'visible';
      video.loop = true;
      video.playbackRate = 0.85;
      video.src = `/api/video/${encodeURIComponent(file)}?v=${Date.now()}`;
      video.onerror = () => {
        $('#output-video-state').textContent = 'تعذر تشغيل الفيديو';
        status.textContent = 'تم إنشاء الإجابة، لكن المتصفح لم يستطع تحميل الفيديو. افتح تتبّع المعالجة للمراجعة.';
      };
      video.onloadeddata = () => {
        $('#output-video-state').textContent = 'إعادة مستمرة · سرعة 0.85×';
        video.play().catch(() => { status.textContent = 'الإجابة جاهزة. اضغط تشغيل لمشاهدة الفيديو.'; });
      };
      video.load();
    }
    $('#input-video-state').textContent = 'تم الفهم';
    status.textContent = 'اكتملت المحادثة بنجاح.';
  } catch (error) {
    status.textContent = error.message;
    $('#output-video-state').textContent = 'تعذر الإنشاء';
  } finally {
    isProcessing = false;
    button.disabled = false;
    button.querySelector('span').textContent = 'إرسال للمساعد';
  }
}

$('#run-conversation').addEventListener('click', runConversation);

$('#translate').addEventListener('click', async () => {
  const text = $('#text').value.trim();
  if (!text) return;
  $('#status').textContent = 'جارٍ التحويل…'; $('#result').innerHTML = '';
  try {
    const response = await fetch('/api/text-to-sign', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text, compose_video:true})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'فشل الطلب');
    $('#result').innerHTML = `<p>${escapeHtml(data.simplified_sentence)}</p>${data.matches.map(x => `<span class="gloss">${escapeHtml(x.gloss || x.query)}</span>`).join('')}`;
    if (data.video_path) {
      const file = data.video_path.split(/[\\/]/).pop();
      $('#result').innerHTML += `<video controls autoplay muted playsinline src="/api/video/${encodeURIComponent(file)}?v=${Date.now()}"></video>`;
    }
    $('#status').textContent = '';
  } catch (error) { $('#status').textContent = error.message; }
});
