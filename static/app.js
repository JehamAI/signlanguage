const button = document.querySelector('#translate');

async function loadRecognitionVocabulary() {
  const container = document.querySelector('#recognition-vocabulary');
  if (!container) return;
  try {
    const response = await fetch('/api/recognition-vocabulary');
    const words = await response.json();
    if (!response.ok) throw new Error('تعذر تحميل القائمة');
    container.textContent = words.map(item => item.arabic).join(' · ');
  } catch (error) {
    container.textContent = 'تعذر تحميل قائمة الكلمات. تأكد من تشغيل الخادم ثم أعد تحميل الصفحة.';
  }
}

loadRecognitionVocabulary();

button.addEventListener('click', async () => {
  const text = document.querySelector('#text').value.trim();
  const status = document.querySelector('#status');
  const result = document.querySelector('#result');
  if (!text) return;
  status.textContent = 'جارٍ التحويل…'; result.innerHTML = '';
  try {
    const response = await fetch('/api/text-to-sign', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text, compose_video:true})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'فشل الطلب');
    result.innerHTML = `<p>${data.simplified_sentence}</p>` + data.matches.map(x => `<span class="gloss">${x.gloss || x.query}</span>`).join('') + (data.review_required ? '<p class="warning">بعض الكلمات تحتاج إلى تهجئة أو مراجعة خبير.</p>' : '');
    if (data.video_path) {
      const file = data.video_path.split(/[\\/]/).pop();
      result.innerHTML += `<video id="sign-video" controls autoplay muted playsinline preload="auto" width="100%"><source src="/api/video/${encodeURIComponent(file)}?v=${Date.now()}" type="video/mp4">Your browser cannot play this video.</video>`;
      const video = document.querySelector('#sign-video');
      video.load();
      video.play().catch(() => { status.textContent = 'تم إنشاء الفيديو. اضغط زر التشغيل لبدء العرض.'; });
    }
    status.textContent = '';
  } catch (error) { status.textContent = error.message; }
});

const conversationButton = document.querySelector('#run-conversation');
const signFileInput = document.querySelector('#sign-files');
const selectedSignFiles = [];

function renderSelectedSignFiles() {
  const container = document.querySelector('#selected-sign-files');
  const continuousToggle = document.querySelector('#continuous-video');
  if (!selectedSignFiles.length) {
    container.textContent = 'لم يتم اختيار مقاطع بعد.';
    return;
  }
  if (selectedSignFiles.length > 1) continuousToggle.checked = false;
  container.innerHTML = selectedSignFiles
    .map((file, index) => `<div>${index + 1}. ${file.name}</div>`)
    .join('');
}

signFileInput.addEventListener('change', () => {
  for (const file of signFileInput.files) {
    const duplicate = selectedSignFiles.some(
      item => item.name === file.name && item.size === file.size && item.lastModified === file.lastModified
    );
    if (!duplicate) selectedSignFiles.push(file);
  }
  // Permit selecting another clip in a later dialog without replacing the accumulated list.
  signFileInput.value = '';
  renderSelectedSignFiles();
});

document.querySelector('#clear-sign-files').addEventListener('click', () => {
  selectedSignFiles.length = 0;
  signFileInput.value = '';
  document.querySelector('#continuous-video').checked = true;
  renderSelectedSignFiles();
});

conversationButton.addEventListener('click', async () => {
  const status = document.querySelector('#conversation-status');
  const result = document.querySelector('#conversation-result');
  if (!selectedSignFiles.length) { status.textContent = 'اختر مقطعاً واحداً على الأقل.'; return; }
  const form = new FormData();
  selectedSignFiles.forEach(file => form.append('files', file));
  const allow = document.querySelector('#allow-low').checked;
  const continuous = document.querySelector('#continuous-video').checked;
  status.textContent = 'جارٍ التعرف على الإشارات وتشغيل RAG…';
  result.innerHTML = '';
  try {
    const response = await fetch(`/api/sign-conversation?allow_low_confidence=${allow}&continuous_video=${continuous}`, {method:'POST', body:form});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'فشل المسار');
    const predictions = data.recognition || data.predictions || [];
    if (continuous) {
      const boundaries = data.segmentation || [];
      result.innerHTML = `<div class="stage"><b>التقسيم الزمني</b><p>تم اكتشاف ${boundaries.length} مقطع/كلمة.</p>${boundaries.map((x, i) => `<p>${i + 1}. ${x.start_seconds.toFixed(2)}s – ${x.end_seconds.toFixed(2)}s</p>`).join('')}</div>`;
      if (boundaries.length < 2) {
        result.innerHTML += '<p class="warning">تم اكتشاف كلمة واحدة فقط. استخدم وقفة محايدة واضحة بين الإشارتين مع خفض اليدين.</p>';
      }
    }
    result.innerHTML += `<div class="stage"><b>1. الكلمات المتعرّف عليها</b><p>${predictions.map(x => `${x.gloss || 'غير معروف'} <span class="score">${(x.confidence * 100).toFixed(1)}%</span>`).join(' ← ')}</p></div>`;
    if (data.status === 'review_required') {
      result.innerHTML += `<p class="warning">${data.message}</p>`;
      status.textContent = 'تحتاج النتيجة إلى مراجعة. يمكنك تفعيل خيار المتابعة التجريبية.';
      return;
    }
    result.innerHTML += `<div class="stage"><b>2. الجملة العربية</b><p>${data.reconstructed_question}</p></div>`;
    result.innerHTML += `<div class="stage"><b>3. معلومات RAG</b>${(data.retrieved_context || []).map(x => `<p>${x.text}</p>`).join('')}</div>`;
    result.innerHTML += `<div class="stage"><b>4. إجابة LLM المقيّدة بالمعلومات</b><p>${data.answer}</p></div>`;
    const outGlosses = (data.answer_glosses && data.answer_glosses.length) ? data.answer_glosses : data.matches.map(x => x.gloss || x.query);
    result.innerHTML += `<div class="stage"><b>5. كلمات الإشارة الناتجة (KArSL-100)</b><p>${outGlosses.map(x => x.gloss || x).join(' ← ')}</p></div>`;
    if (data.video_path) {
      const file = data.video_path.split(/[\\/]/).pop();
      result.innerHTML += `<div class="stage"><b>6. فيديو الإجابة</b><video id="conversation-video" controls autoplay muted playsinline preload="auto" width="100%"><source src="/api/video/${encodeURIComponent(file)}?v=${Date.now()}" type="video/mp4"></video></div>`;
      const video = document.querySelector('#conversation-video');
      video.load(); video.play().catch(() => {});
    }
    status.textContent = 'اكتمل المسار.';
  } catch (error) { status.textContent = error.message; }
});
