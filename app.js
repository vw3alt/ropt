// config
const CONFIG = {
  model: "Llama-3.2-1B-Instruct-q4f16_1-MLC", 
  topK: 3,
  embeddingModel: "Xenova/all-MiniLM-L6-v2",
  defaultPersona: `You are a human named ropt. You are chatting on Discord.
  
Rules for your behavior:
1. NEVER break character. Never mention you are an AI, an assistant, or a clone, because you're not - you're only ropt.  If you're asked who you are, you say "I am ropt".
2. Be brief. Write 1 to 2 short sentences maximum. No need to be formal, this is a conversation.
3. Sprinkle in some absurdist humor.  The goal is for your conversations to be somewhat nonsensical.
4. Never flirt with the user unless they flirt with you.  If the user is flirty to you, flirt aggressively back.`,
};

let engine         = null;
let embedder       = null;
let kbChunks       = [];
let kbVectors      = [];
let conversationHistory = [];
let persona        = CONFIG.defaultPersona;
let isGenerating   = false;

// ui
const $  = id => document.getElementById(id);
const statusDot  = $('status-dot');
const statusText = $('status-text');
const progressWrap = $('progress-wrap');
const progressBar  = $('progress-bar');
const chat         = $('chat');
const userInput    = $('user-input');
const sendBtn      = $('send-btn');

function setStatus(state, text) {
  statusDot.className = state;
  statusText.textContent = text;
}

function setProgress(pct) {
  if (pct === null) {
    progressWrap.style.display = 'none';
    return;
  }
  progressWrap.style.display = 'block';
  progressBar.style.width = pct + '%';
}

function appendMessage(role, text, ragChunks = null) {
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;

  const label = document.createElement('div');
  label.className = 'msg-label';
  label.textContent = role === 'user' ? 'you' : 'ropt';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.textContent = text;

  wrap.appendChild(label);
  wrap.appendChild(bubble);

  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
  return bubble;
}

$('close-popup').addEventListener('click', () => {
  $('setup-popup').style.display = 'none';
});

// I couldn't be bothered and just did cosine similarity
function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot   += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-10);
}

function topK(queryVec, k) {
  if (kbVectors.length === 0) return [];

  const scored = kbVectors.map((vec, i) => ({
    score: cosineSimilarity(queryVec, vec),
    chunk: kbChunks[i],
  }));

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k).map(s => s.chunk);
}

// embedding
async function loadEmbedder() {
  if (embedder) return;
  setStatus('loading', 'loading embedding model...');

  const { pipeline, env } = await import(
    'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js'
  );
  env.allowLocalModels = false;

  embedder = await pipeline('feature-extraction', CONFIG.embeddingModel, {
    quantized: true,
  });
  setStatus('ready', 'ready');
}

async function embedQuery(text) {
  const output = await embedder(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

// kb autoloader
async function loadKnowledgeBase() {
  $('kb-status').textContent = 'loading knowledge base...';

  try {
    const response = await fetch('./knowledge-base-embedded.json');
    if (!response.ok) {
      throw new Error(`Failed to fetch file: ${response.status} ${response.statusText}`);
    }
    
    const raw = await response.json();

    if (!Array.isArray(raw) || raw.length === 0) {
      throw new Error('JSON must be a non-empty array');
    }

    const hasPrecomputed = raw[0].embedding && Array.isArray(raw[0].embedding);

    if (hasPrecomputed) {
      kbChunks  = raw.map(({ text, source, date }) => ({ text, source, date }));
      kbVectors = raw.map(r => r.embedding);
      $('kb-status').textContent = `${kbChunks.length} chunks loaded automatically ✓`;

      await loadEmbedder();
    } else {
      $('kb-status').textContent = 'error: file is missing pre-computed embeddings.';
      return;
    }

    setStatus('ready', 'ready — RAG active');
  } catch (err) {
    $('kb-status').textContent = `error: ${err.message}`;
    console.error(err);
  }
}

// rag
async function retrieve(query) {
  if (kbVectors.length === 0) return [];
  const queryVec = await embedQuery(query);
  return topK(queryVec, CONFIG.topK);
}

function buildSystemPrompt(ragChunks) {
  let prompt = persona;

  if (ragChunks.length > 0) {
    const context = ragChunks
      .map(c => `[Context message]: ${c.text}`)
      .join('\n');

    prompt += `\n\n=== CONTEXT ===\nHere are some of your actual past messages. Use these to answer the user's prompt, but MORE IMPORTANTLY, strictly copy the tone, length, and vocabulary of these messages:\n\n${context}\n=== END CONTEXT ===`;
  }

  return prompt;
}

// big boy webllm stuff
async function initModel() {
  setStatus('loading', 'loading WebLLM...');

  const { CreateMLCEngine } = await import(
    'https://esm.run/@mlc-ai/web-llm@0.2.73'
  );

  setProgress(0);
  engine = await CreateMLCEngine(CONFIG.model, {
    initProgressCallback: (p) => {
      const pct = Math.round((p.progress || 0) * 100);
      setProgress(pct);
      setStatus('loading', p.text || `loading model... ${pct}%`);
    },
  });

  setProgress(null);
  setStatus('ready', `${CONFIG.model}`);
  document.getElementById('ready-or-not').textContent = 'ready';
  userInput.disabled = false;
  sendBtn.disabled   = false;
  userInput.focus();
}

let interjections = [];

async function loadInterjections() {
  try {
    const response = await fetch('./interjections.json');
    const data = await response.json();
    interjections = data.phrases;
  } catch (err) {
    console.error("Failed to load interjections:", err);
  }
}

async function simulateStreaming(text, element) {
  const words = text.split(' ');
  let currentText = '';
  
  for (const word of words) {
    currentText += (currentText ? ' ' : '') + word;
    element.textContent = currentText;
    
    await new Promise(resolve => setTimeout(resolve, 900 + Math.random() * 100));
    
    const chat = document.getElementById('chat');
    chat.scrollTop = chat.scrollHeight;
  }
}

async function generate(userMessage) {
  isGenerating = true;
  userInput.disabled = true;
  sendBtn.disabled   = true;

  const roll = Math.random();
  let preSelected = null;
  let cumulativeProb = 0;

  if (conversationHistory.length > 0) {
    for (const item of interjections) {
      cumulativeProb += item.probability;
      if (roll < cumulativeProb) {
        preSelected = item.text;
        break;
      }
    }
  }

  appendMessage('user', userMessage);

  let fullResponse = '';
  let ragChunks = [];

  if (preSelected) {
    fullResponse = preSelected;
    
    const botBubble = appendMessage('bot', '');
    botBubble.classList.add('streaming');
    await new Promise(r => setTimeout(r, 800 + Math.random() * 1500));
    await simulateStreaming(fullResponse, botBubble);
    
    botBubble.classList.remove('streaming');
  } else {
    ragChunks = await retrieve(userMessage);
    const systemPrompt = buildSystemPrompt(ragChunks);
    const messages = [
      { role: 'system', content: systemPrompt },
      ...conversationHistory,
      { role: 'user', content: userMessage },
    ];

    const botBubble = appendMessage('bot', '', ragChunks);
    botBubble.classList.add('streaming');

    try {
      const stream = await engine.chat.completions.create({
        messages,
        stream: true,
        temperature: 0.85,
        max_tokens: 60,
        frequency_penalty: 0.5,
      });

      for await (const chunk of stream) {
        const delta = chunk.choices[0]?.delta?.content || '';
        fullResponse += delta;
        
        const cleanDisplay = fullResponse
          .replace(/\*.*?\*/g, '')
          .replace(/\*/g, '')
          .replace(/\./g, '');

        botBubble.textContent = cleanDisplay;
        chat.scrollTop = chat.scrollHeight;
      }
      
      fullResponse = botBubble.textContent;

    } catch (err) {
      botBubble.textContent = `[error: ${err.message}]`;
    }
    botBubble.classList.remove('streaming');
  }

  conversationHistory.push({ role: 'user', content: userMessage });
  conversationHistory.push({ role: 'assistant', content: fullResponse });
  
  if (conversationHistory.length > 20) conversationHistory = conversationHistory.slice(-20);

  isGenerating = false;
  userInput.disabled = false;
  sendBtn.disabled = false;
  userInput.focus();
}

// ui
async function handleSend() {
  if (isGenerating || !engine) return;
  const text = userInput.value.trim();
  if (!text) return;
  userInput.value = '';
  userInput.style.height = '44px';
  await generate(text);
}

sendBtn.addEventListener('click', handleSend);

userInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
});

userInput.addEventListener('input', () => {
  userInput.style.height = '44px';
  userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
});

// boot
loadKnowledgeBase();
loadInterjections();

initModel().catch(err => {
  setStatus('error', `failed to load model: ${err.message}`);
  console.error(err);
});