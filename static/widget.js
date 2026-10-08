(function () {
  const script = document.currentScript;
  if (!script) return;

  const projectId = script.dataset.project;
  if (!projectId) return;

  const apiBase = new URL(script.src).origin;

  // Session-based userId — used only for lead-capture dedup (see
  // /public/leads below), NOT the actual conversation session.
  let userId = localStorage.getItem("rag_user_id");
  if (!userId) {
    userId = crypto.randomUUID();
    localStorage.setItem("rag_user_id", userId);
  }

  // FIX: this widget used to send `userId` as if it were the chat session,
  // but the backend's /public/chat only ever recognizes a field called
  // `sessionId` — the mismatch meant every single message silently started
  // a brand-new, memoryless chat. Persisted here the same way
  // PublicChatClient.js's shareable-link chat already does (3-hour TTL),
  // and sent back on every call from here on.
  const SESSION_TTL_MS = 3 * 60 * 60 * 1000;
  let sessionId = null;
  try {
    const stored = JSON.parse(localStorage.getItem(`chat_session_${projectId}`) || "null");
    if (stored && stored.expiresAt > Date.now()) sessionId = stored.value;
  } catch (e) {}

  function saveSessionId(id) {
    sessionId = id;
    localStorage.setItem(`chat_session_${projectId}`, JSON.stringify({
      value: id,
      expiresAt: Date.now() + SESSION_TTL_MS,
    }));
  }

  // Lead state
  let lead = JSON.parse(localStorage.getItem(`chat_lead_${projectId}`) || "null");
  let awaitingLead = false;
  let pendingQuestion = null;
  let userMessageCount = 0;

  // Lead capture config (fetched from backend)
  let leadConfig = null;

  // Password-protected projects mint a short-lived access token on
  // verification. The widget had no password flow at all: it never sent a
  // token, so a protected project answered 401 to every message and the
  // visitor saw a generic error forever with no way to unlock. The
  // shareable-link page has had this flow all along.
  const ACCESS_KEY = `chat_access_${projectId}`;
  let accessToken = null;
  try { accessToken = localStorage.getItem(ACCESS_KEY); } catch (e) {}

  function saveAccessToken(token) {
    accessToken = token || null;
    try {
      if (token) localStorage.setItem(ACCESS_KEY, token);
      else localStorage.removeItem(ACCESS_KEY);
    } catch (e) {}
  }

  const history = [];

  // Quotes matter as much as angle brackets here: the linkifier below puts
  // the matched text inside href="...", so a URL containing a double quote
  // used to break out of the attribute and inject an event handler running
  // on the MERCHANT'S own origin. Reachable through poisoned knowledge-base
  // content, a scraped website source, or plain prompt injection.
  function esc(text) {
    return (text || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // Stops at quotes as well as whitespace/angle brackets, so a trailing
  // quote can never land inside the attribute in the first place.
  const URL_RE = /(https?:\/\/[^\s<>"']+)/g;

  function render(text) {
    const escaped = esc(text);
    // Auto-linkify so a checkout/shop link the bot relays is actually
    // tappable — this widget only ever renders plain escaped text, unlike
    // WhatsApp, which auto-links URLs on its own.
    const linked = escaped.replace(URL_RE, url => {
      if (!/^https?:\/\//i.test(url)) return url;
      return `<a href="${url}" target="_blank" rel="noopener noreferrer" style="color:#2563eb;text-decoration:underline">${url}</a>`;
    });
    return linked.replace(/\n/g, "<br/>");
  }

  // ---------------- FETCH LEAD CONFIG ----------------
  async function fetchLeadConfig() {
    try {
      const res = await fetch(`${apiBase}/public/lead-config/${projectId}`);
      const data = await res.json();
      leadConfig = data.enabled ? data : null;
    } catch (e) {
      leadConfig = null;
    }
  }

  // ---------------- RESTORE HISTORY ----------------
  // Redraws earlier messages from this session so the widget's screen
  // matches what the bot actually remembers (see sessionId persistence
  // above) — without this, reopening the bubble or reloading the page
  // showed an empty box even though the backend recalled everything.
  async function restoreHistory() {
    if (!sessionId) { markRestored(); return; }
    try {
      const res = await fetch(`${apiBase}/public/chat/history/${sessionId}?project_id=${encodeURIComponent(projectId)}`);
      const data = await res.json();
      for (const m of (data.messages || [])) {
        var text = m.content || "";
        if (text.indexOf("[Human] ") === 0) text = text.slice(8);  // team reply marker
        addMsg(m.role === "user" ? "user" : "assistant", render(text), false);
      }
      if (data.messages && data.messages.length) userMessageCount = data.messages.filter(m => m.role === "user").length;
      // Reloaded while a person was handling the chat: keep listening for
      // their replies.
      if (data.messages && data.messages.some(m => (m.content || "").indexOf("[Human] ") === 0 ||
          (m.content || "").indexOf("Connecting you to our team") === 0)) startHumanPoll();
    } catch (e) {}
    markRestored();
  }

  // ---------------- UI ----------------
  // Everything lives in a shadow root. The host website's CSS can't restyle
  // the widget (a site-wide `button {…}` or `input {…}` rule used to), the
  // widget's styles can't leak into the site, and it can use a real
  // stylesheet — hover, focus and @keyframes animation, none of which inline
  // styles can express.
  const STYLE = `
    :host {
      all: initial;
      --c1: #6366f1; --c2: #8b5cf6; --c3: #d946ef;
      --glow: rgba(99,102,241,.45); --ring: rgba(139,92,246,.2);
      --grad: linear-gradient(135deg, var(--c1), var(--c2) 55%, var(--c3));
    }
    *, *::before, *::after { box-sizing: border-box; }
    button, input { font: inherit; }

    /* ---- launcher ---- */
    .dock { position: fixed; right: 20px; bottom: 20px; width: 60px; height: 60px; z-index: 2147483000; }
    .ring { position: absolute; inset: 0; border-radius: 50%; background: var(--grad);
            animation: zv-ring 2.8s ease-out infinite; pointer-events: none; }
    .dock.open .ring { display: none; }
    .launcher { position: relative; width: 60px; height: 60px; border: none; border-radius: 50%; cursor: pointer;
                color: #fff; display: flex; align-items: center; justify-content: center; padding: 0;
                background: var(--grad); background-size: 200% 200%;
                box-shadow: 0 10px 28px var(--glow), 0 2px 8px rgba(0,0,0,.16);
                animation: zv-pop .55s cubic-bezier(.34,1.56,.64,1) .25s both, zv-flow 9s ease infinite;
                transition: transform .3s cubic-bezier(.34,1.56,.64,1), box-shadow .3s; }
    .launcher:hover { transform: scale(1.1) rotate(-6deg); box-shadow: 0 14px 34px var(--glow), 0 3px 10px rgba(0,0,0,.2); }
    .launcher:active { transform: scale(.94); }
    .launcher:focus-visible, .close:focus-visible, .send:focus-visible { outline: 3px solid #fff; outline-offset: 2px; box-shadow: 0 0 0 5px var(--c2); }
    .launcher svg { position: absolute; width: 27px; height: 27px; transition: transform .4s cubic-bezier(.34,1.56,.64,1), opacity .2s; }
    .ico-close { opacity: 0; transform: rotate(-90deg) scale(.4); }
    .dock.open .ico-chat { opacity: 0; transform: rotate(90deg) scale(.4); }
    .dock.open .ico-close { opacity: 1; transform: none; }

    /* ---- panel ---- */
    .panel { position: fixed; right: 20px; bottom: 92px; width: 380px; max-width: calc(100vw - 24px);
             height: min(620px, calc(100vh - 116px)); height: min(620px, calc(100dvh - 116px));
             display: flex; flex-direction: column; background: #fff; border-radius: 24px; overflow: hidden; z-index: 2147483000;
             font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
             color: #1f2937; -webkit-font-smoothing: antialiased;
             box-shadow: 0 28px 80px rgba(49,46,129,.32), 0 4px 18px rgba(0,0,0,.08);
             opacity: 0; visibility: hidden; pointer-events: none;
             transform: translateY(18px) scale(.93); transform-origin: bottom right;
             transition: opacity .25s ease, transform .4s cubic-bezier(.34,1.3,.64,1), visibility 0s linear .4s; }
    .panel.open { opacity: 1; visibility: visible; pointer-events: auto; transform: none;
                  transition: opacity .25s ease, transform .4s cubic-bezier(.34,1.3,.64,1), visibility 0s; }

    /* ---- header ---- */
    .head { position: relative; display: flex; align-items: center; gap: 12px; padding: 18px 16px 20px; color: #fff; overflow: hidden;
            background: var(--grad); background-size: 200% 200%; animation: zv-flow 10s ease infinite; }
    .head::before, .head::after { content: ""; position: absolute; border-radius: 50%; pointer-events: none; }
    .head::before { width: 150px; height: 150px; right: -40px; top: -70px; background: rgba(255,255,255,.14); }
    .head::after  { width: 90px; height: 90px; right: 70px; bottom: -55px; background: rgba(255,255,255,.1); }
    .head > * { position: relative; z-index: 1; }
    .avatar-wrap { position: relative; flex: none; }
    .avatar { width: 44px; height: 44px; border-radius: 50%; overflow: hidden; display: flex; align-items: center; justify-content: center;
              background: rgba(255,255,255,.22); border: 2px solid rgba(255,255,255,.6); }
    .avatar img { width: 100%; height: 100%; object-fit: cover; display: block; }
    .avatar svg { width: 22px; height: 22px; }
    .live { position: absolute; right: -1px; bottom: -1px; width: 12px; height: 12px; border-radius: 50%; background: #22c55e; border: 2px solid #fff; }
    .live::after { content: ""; position: absolute; inset: -2px; border-radius: 50%; border: 2px solid #22c55e; animation: zv-ping 2s ease-out infinite; }
    .titles { flex: 1; min-width: 0; }
    .title { font-weight: 650; font-size: 16px; line-height: 1.2; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .sub { font-size: 12px; opacity: .9; margin-top: 3px; }
    .close { flex: none; width: 34px; height: 34px; border: none; border-radius: 50%; cursor: pointer; padding: 0; color: #fff;
             background: rgba(255,255,255,.2); display: flex; align-items: center; justify-content: center;
             transition: background .2s, transform .3s cubic-bezier(.34,1.56,.64,1); }
    .close:hover { background: rgba(255,255,255,.34); transform: rotate(90deg); }
    .close svg { width: 16px; height: 16px; }

    /* ---- messages ---- */
    .msgs { position: relative; flex: 1; overflow-y: auto; padding: 18px 14px 8px; font-size: 14px; line-height: 1.5;
            background: linear-gradient(180deg, #f3f0ff 0%, #f9f8ff 38%, #fff 100%); }
    .msgs::-webkit-scrollbar { width: 6px; }
    .msgs::-webkit-scrollbar-thumb { background: #d6d3ee; border-radius: 3px; }
    .row { display: flex; margin-bottom: 10px; animation: zv-in .4s cubic-bezier(.22,1,.36,1) both; }
    .row.still { animation: none; }
    .row.user { justify-content: flex-end; }
    .bubble { max-width: 84%; padding: 10px 14px; border-radius: 18px; overflow-wrap: anywhere; }
    .row.assistant .bubble { background: #fff; color: #1f2937; border: 1px solid #ebe9f7; border-bottom-left-radius: 6px;
                             box-shadow: 0 2px 8px rgba(79,70,229,.07); }
    .row.user .bubble { background: var(--grad); color: #fff; border-bottom-right-radius: 6px;
                        box-shadow: 0 6px 16px var(--glow); }
    .row.user .bubble a { color: #fff !important; }
    .typing .bubble { display: flex; align-items: center; gap: 5px; padding: 14px 16px; }
    .typing .bubble i { width: 7px; height: 7px; border-radius: 50%; background: var(--grad); animation: zv-bounce 1.2s ease-in-out infinite; }
    .typing .bubble i:nth-child(2) { animation-delay: .15s; }
    .typing .bubble i:nth-child(3) { animation-delay: .3s; }

    /* ---- composer ---- */
    .composer { display: flex; align-items: center; gap: 10px; padding: 12px 14px 14px; background: #fff; border-top: 1px solid #efedf8; }
    .field { flex: 1; min-width: 0; background: #f4f3fb; border: 1.5px solid transparent; border-radius: 26px; padding: 0 16px;
             transition: border-color .2s, box-shadow .2s, background .2s; }
    .field:focus-within { background: #fff; border-color: var(--c2); box-shadow: 0 0 0 4px var(--ring); }
    .input { width: 100%; height: 42px; border: none; outline: none; background: transparent; font-size: 14px; color: #1f2937; }
    .input::placeholder { color: #9ca3af; }
    .input:disabled { cursor: not-allowed; }
    .send { flex: none; width: 42px; height: 42px; border: none; border-radius: 50%; cursor: pointer; padding: 0; color: #fff;
            display: flex; align-items: center; justify-content: center; background: var(--grad);
            box-shadow: 0 6px 16px var(--glow);
            transition: transform .25s cubic-bezier(.34,1.56,.64,1), box-shadow .25s, opacity .2s; }
    .send:hover:not(:disabled) { transform: scale(1.1) rotate(-8deg); }
    .send:active:not(:disabled) { transform: scale(.92); }
    .send:disabled { cursor: not-allowed; box-shadow: none; }
    .send svg { width: 19px; height: 19px; margin-left: -1px; }

    /* ---- in-chat cards (lead form, password) ---- */
    .bubble.card { width: 92%; max-width: 92%; padding: 16px; border-radius: 18px; }
    .card-title { font-weight: 650; font-size: 14.5px; margin-bottom: 3px; }
    .card-sub { font-size: 12.5px; color: #6b7280; margin-bottom: 12px; }
    .fld { width: 100%; height: 40px; margin-bottom: 8px; padding: 0 13px; border: 1.5px solid #e5e3f3; border-radius: 12px;
           background: #fafaff; color: #1f2937; font-size: 13.5px; outline: none; transition: border-color .2s, box-shadow .2s, background .2s; }
    .fld:focus { background: #fff; border-color: var(--c2); box-shadow: 0 0 0 4px var(--ring); }
    .err { color: #dc2626; font-size: 12px; margin: 0 0 8px; }
    .cta { width: 100%; height: 42px; border: none; border-radius: 12px; cursor: pointer; color: #fff; font-weight: 600; font-size: 14px;
           background: var(--grad); box-shadow: 0 6px 16px var(--glow);
           transition: transform .2s cubic-bezier(.34,1.56,.64,1), box-shadow .2s, opacity .2s; }
    .cta:hover:not(:disabled) { transform: translateY(-1px) scale(1.01); }
    .cta:active:not(:disabled) { transform: scale(.98); }
    .cta:disabled { opacity: .6; cursor: default; }

    @keyframes zv-pop    { from { opacity: 0; transform: scale(0) rotate(-40deg); } to { opacity: 1; transform: none; } }
    @keyframes zv-ring   { 0% { transform: scale(1); opacity: .5; } 100% { transform: scale(1.75); opacity: 0; } }
    @keyframes zv-flow   { 0%, 100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
    @keyframes zv-in     { from { opacity: 0; transform: translateY(12px) scale(.96); } to { opacity: 1; transform: none; } }
    @keyframes zv-ping   { 0% { transform: scale(1); opacity: .8; } 100% { transform: scale(2.1); opacity: 0; } }
    @keyframes zv-bounce { 0%, 60%, 100% { transform: translateY(0); opacity: .45; } 30% { transform: translateY(-6px); opacity: 1; } }

    @media (max-width: 480px) {
      .dock { right: 14px; bottom: 14px; }
      .panel { left: 12px; right: 12px; width: auto; max-width: none; bottom: 86px; border-radius: 20px;
               height: min(620px, calc(100vh - 104px)); height: min(620px, calc(100dvh - 104px)); }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { animation: none !important; transition-duration: .01ms !important; }
    }
  `;

  const host = document.createElement("div");
  host.id = "zavo-chat-widget";
  const root = host.attachShadow({ mode: "open" });

  root.innerHTML = `
    <style>${STYLE}</style>

    <div class="dock" id="dock">
      <span class="ring"></span>
      <button class="launcher" id="launcher" type="button" aria-label="Open chat" aria-expanded="false" aria-controls="panel">
        <svg class="ico-chat" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.4 8.4 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.4 8.4 0 0 1-3.8-.9L3 21l1.9-5.7a8.4 8.4 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.4 8.4 0 0 1 3.8-.9h.5a8.5 8.5 0 0 1 8 8z"/></svg>
        <svg class="ico-close" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M18 6 6 18M6 6l12 12"/></svg>
      </button>
    </div>

    <section class="panel" id="panel" role="dialog" aria-label="Chat" aria-hidden="true">
      <header class="head">
        <div class="avatar-wrap">
          <div class="avatar" id="avatar">
            <svg viewBox="0 0 24 24" fill="#fff"><path d="M12 2l1.9 5.9L20 10l-6.1 2.1L12 18l-1.9-5.9L4 10l6.1-2.1z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z" opacity=".85"/></svg>
          </div>
          <span class="live"></span>
        </div>
        <div class="titles">
          <div class="title" id="title">Chat with us</div>
          <div class="sub">Online &middot; replies instantly</div>
        </div>
        <button class="close" id="closeBtn" type="button" aria-label="Close chat">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M18 6 6 18M6 6l12 12"/></svg>
        </button>
      </header>

      <div class="msgs" id="msgs" role="log" aria-live="polite"></div>

      <form class="composer" id="chatForm" autocomplete="off">
        <div class="field">
          <input class="input" id="chat-input" placeholder="Type your question..." maxlength="4000" autocomplete="off" aria-label="Your message"/>
        </div>
        <button class="send" id="send-btn" type="submit" aria-label="Send">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 2 11 13"/><path d="M22 2 15 22l-4-9-9-4z"/></svg>
        </button>
      </form>
    </section>
  `;
  document.body.appendChild(host);

  const dock = root.getElementById("dock");
  const launcher = root.getElementById("launcher");
  const box = root.getElementById("panel");
  const closeBtn = root.getElementById("closeBtn");
  const msgs = root.getElementById("msgs");
  const form = root.getElementById("chatForm");
  const input = root.getElementById("chat-input");
  const sendBtn = root.getElementById("send-btn");

  // ---- merchant branding (name, logo, brand colour) ----
  // A brand colour becomes the gradient; near-black, near-white and grey
  // colours would only make a dull one, so those keep the default palette.
  function applyBrand(hex) {
    if (!/^#[0-9a-f]{6}$/i.test(hex || "")) return;
    const r = parseInt(hex.slice(1, 3), 16) / 255;
    const g = parseInt(hex.slice(3, 5), 16) / 255;
    const b = parseInt(hex.slice(5, 7), 16) / 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min;
    const l = (max + min) / 2;
    const s = d === 0 ? 0 : d / (1 - Math.abs(2 * l - 1));
    if (l < 0.12 || l > 0.9 || s < 0.12) return;
    let h = 0;
    if (d !== 0) {
      if (max === r) h = ((g - b) / d) % 6;
      else if (max === g) h = (b - r) / d + 2;
      else h = (r - g) / d + 4;
      h = (h * 60 + 360) % 360;
    }
    const hsl = (hh, ss, ll, a) =>
      `hsl(${Math.round(hh)} ${Math.round(ss * 100)}% ${Math.round(ll * 100)}%${a == null ? "" : " / " + a})`;
    host.style.setProperty("--c1", hsl(h, s, Math.max(l - 0.07, 0.18)));
    host.style.setProperty("--c2", hsl(h, s, l));
    host.style.setProperty("--c3", hsl((h + 32) % 360, Math.min(s + 0.05, 1), Math.min(l + 0.1, 0.72)));
    host.style.setProperty("--glow", hsl(h, s, l, 0.45));
    host.style.setProperty("--ring", hsl(h, s, l, 0.2));
  }

  async function fetchWidgetConfig() {
    try {
      const res = await fetch(`${apiBase}/public/widget-config/${projectId}`);
      if (!res.ok) return;
      const c = await res.json();
      applyBrand(c.brand_color);
      if (c.name) {
        root.getElementById("title").textContent = c.name;
        box.setAttribute("aria-label", "Chat with " + c.name);
      }
      // Set through .src on a created element, never innerHTML, and only for
      // https: the value comes from the merchant's settings.
      if (typeof c.logo_url === "string" && /^https:\/\//i.test(c.logo_url)) {
        const img = document.createElement("img");
        img.alt = "";
        img.onload = () => { root.getElementById("avatar").replaceChildren(img); };
        img.src = c.logo_url;
      }
    } catch (e) {}
  }

  let hasOpened = false;
  function setOpen(open) {
    dock.classList.toggle("open", open);
    box.classList.toggle("open", open);
    launcher.setAttribute("aria-expanded", String(open));
    launcher.setAttribute("aria-label", open ? "Close chat" : "Open chat");
    box.setAttribute("aria-hidden", String(!open));
    if (!open) {
      if (flowHost.plugin) flowHost.plugin.onClose();
      return;
    }
    // Shown once, the first time the widget is opened — gives the visitor
    // a hint of what to ask instead of a blank box. Skipped if earlier
    // messages were just restored (see restoreHistory below), and not
    // shown again on later opens/closes so it doesn't repeat above real
    // conversation.
    // A website flow, when one is active, opens with its own first message
    // instead of this generic greeting.
    if (!hasOpened && !msgs.children.length && !(flowHost.plugin && flowHost.plugin.onOpen())) {
      addMsg("assistant", "&#128075; Hi! Ask me anything &mdash; I'm here to help.");
    }
    hasOpened = true;
    msgs.scrollTop = msgs.scrollHeight;
    // Focus only where there is a physical keyboard: on a phone it would pop
    // the keyboard over the conversation the moment the panel opens.
    if (window.matchMedia && window.matchMedia("(pointer: fine)").matches) {
      setTimeout(() => { if (!input.disabled) input.focus(); }, 300);
    }
  }
  launcher.onclick = () => setOpen(!box.classList.contains("open"));
  closeBtn.onclick = () => { setOpen(false); launcher.focus(); };
  root.addEventListener("keydown", (e) => { if (e.key === "Escape" && box.classList.contains("open")) setOpen(false); });

  function scrollToEnd(animate) {
    if (animate && msgs.scrollTo) msgs.scrollTo({ top: msgs.scrollHeight, behavior: "smooth" });
    else msgs.scrollTop = msgs.scrollHeight;
  }

  // `html` is already escaped by render() (or is one of our own strings).
  // animate=false for restored history, so a long conversation doesn't
  // replay its entrance animation all at once.
  function addMsg(role, html, animate = true) {
    const el = document.createElement("div");
    el.className = "row " + (role === "user" ? "user" : "assistant") + (animate ? "" : " still");
    el.innerHTML = `<div class="bubble">${html}</div>`;
    msgs.appendChild(el);
    scrollToEnd(animate);
  }

  function showTyping() {
    const el = document.createElement("div");
    el.className = "row assistant typing";
    el.setAttribute("aria-label", "Typing");
    el.innerHTML = `<div class="bubble"><i></i><i></i><i></i></div>`;
    msgs.appendChild(el);
    scrollToEnd(true);
    return el;
  }

  function blockInput(placeholder) {
    input.disabled = true;
    input.placeholder = placeholder || "Please fill the form to continue...";
    sendBtn.disabled = true;
    sendBtn.style.opacity = "0.5";
  }

  function unblockInput() {
    input.disabled = false;
    input.placeholder = "Type your question...";
    sendBtn.disabled = false;
    sendBtn.style.opacity = "1";
    if (box.classList.contains("open") && window.matchMedia && window.matchMedia("(pointer: fine)").matches) input.focus();
  }

  // ---------------- LEAD FORM ----------------
  // Password unlock. Mirrors the shareable-link page's flow: verify once,
  // keep the returned short-lived token, replay it on every message.
  function showPasswordForm() {
    blockInput();

    const existing = root.getElementById("pw-overlay");
    if (existing) existing.remove();

    const overlay = document.createElement("div");
    overlay.id = "pw-overlay";
    overlay.className = "row assistant";
    overlay.innerHTML = `
      <div class="bubble card">
        <div class="card-title">&#128274; This chat is protected</div>
        <div class="card-sub">Enter the password to continue.</div>
        <input id="pw-input" class="fld" type="password" placeholder="Password" autocomplete="current-password"/>
        <div id="pw-error" class="err" style="display:none"></div>
        <button id="pw-submit" class="cta" type="button">Unlock</button>
      </div>
    `;
    msgs.appendChild(overlay);
    scrollToEnd(true);

    const input = overlay.querySelector("#pw-input");
    const errorEl = overlay.querySelector("#pw-error");
    const submitBtn = overlay.querySelector("#pw-submit");

    input.focus();

    async function submit() {
      const password = input.value;
      if (!password) return;

      errorEl.style.display = "none";
      submitBtn.textContent = "Checking...";
      submitBtn.disabled = true;

      try {
        const res = await fetch(`${apiBase}/public/chat/verify-password`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ projectId, password }),
        });
        const data = await res.json().catch(() => ({}));

        if (!res.ok || !data.accessToken) {
          // One message for a wrong password and for a rate limit, so this
          // is not an oracle for whether a password is set.
          errorEl.textContent = res.status === 429
            ? "Too many attempts. Please wait a moment."
            : "Incorrect password.";
          errorEl.style.display = "block";
          submitBtn.textContent = "Unlock";
          submitBtn.disabled = false;
          return;
        }

        saveAccessToken(data.accessToken);
        overlay.remove();
        unblockInput();

        const q = pendingQuestion;
        pendingQuestion = null;
        if (q) askBot(q);
      } catch (e) {
        errorEl.textContent = "Could not reach the server. Please try again.";
        errorEl.style.display = "block";
        submitBtn.textContent = "Unlock";
        submitBtn.disabled = false;
      }
    }

    submitBtn.addEventListener("click", submit);
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") submit(); });
  }

  function showLeadForm() {
    awaitingLead = true;
    blockInput();

    // Set by an authenticated project member and interpolated into innerHTML
    // below, so it runs on every site the merchant embeds this widget on.
    // esc() is what makes that safe; length is capped server-side too.
    const title = esc(leadConfig?.form_title || "Before we continue...");
    const subtitle = esc(leadConfig?.form_subtitle || "Please share your details to keep chatting.");

    // Already showing: just bring it back into view rather than stacking a
    // second form.
    const showing = root.getElementById("lead-form-card");
    if (showing) {
      msgs.scrollTop = msgs.scrollHeight;
      return;
    }

    // Part of the conversation, in the same left-aligned bubble style as the
    // bot's messages, right after the question that triggered it. It used to
    // be a full-area overlay pinned to the top of the scrolling message list:
    // it hid the chat, and once the chat had scrolled it sat off-screen
    // above the visible area.
    const overlay = document.createElement("div");
    overlay.id = "lead-form-card";
    overlay.className = "row assistant";

    overlay.innerHTML = `
      <div class="bubble card">
        <div class="card-title">${title}</div>
        <div class="card-sub">${subtitle}</div>
        <input id="lf-name" class="fld" type="text" placeholder="Your name *" autocomplete="name"/>
        <input id="lf-email" class="fld" type="email" placeholder="Email address *" autocomplete="email"/>
        <input id="lf-phone" class="fld" type="tel" placeholder="Phone number *" autocomplete="tel"/>
        <div id="lf-error" class="err" style="display:none"></div>
        <button id="lf-submit" class="cta" type="button">Continue chatting &rarr;</button>
      </div>
    `;

    msgs.appendChild(overlay);
    scrollToEnd(true);

    // Submit handler
    overlay.querySelector("#lf-submit").onclick = async () => {
      const name = overlay.querySelector("#lf-name").value.trim();
      const email = overlay.querySelector("#lf-email").value.trim();
      const phone = overlay.querySelector("#lf-phone").value.trim();
      const errorEl = overlay.querySelector("#lf-error");
      const submitBtn = overlay.querySelector("#lf-submit");

      // Validate
      if (!name || !email || !phone) {
        errorEl.textContent = "All fields are required.";
        errorEl.style.display = "block";
        return;
      }
      if (!email.includes("@")) {
        errorEl.textContent = "Enter a valid email address.";
        errorEl.style.display = "block";
        return;
      }
      if (phone.replace(/\D/g, "").length < 7) {
        errorEl.textContent = "Enter a valid phone number.";
        errorEl.style.display = "block";
        return;
      }

      errorEl.style.display = "none";
      submitBtn.textContent = "Saving...";
      submitBtn.disabled = true;

      try {
        const res = await fetch(`${apiBase}/public/leads`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            project_id: projectId,
            session_id: userId,
            // The real conversation, verified server-side against the chats
            // table. Without it /public/leads was an unauthenticated write
            // into any merchant's contact list.
            chat_session_id: sessionId,
            name,
            email,
            phone,
          }),
        });

        if (res.ok) {
          lead = { name, email, phone };
          localStorage.setItem(`chat_lead_${projectId}`, JSON.stringify(lead));

          awaitingLead = false;
          overlay.remove();
          unblockInput();

          // A waiting question is answered right below, so a thank-you is all
          // that's needed; only offer help when there is nothing pending.
          addMsg("assistant", pendingQuestion
            ? `Thanks ${esc(name)}!`
            : `Thanks ${esc(name)}! How can I help you?`);

          // Answer the question they had asked before form appeared
          if (pendingQuestion) {
            await askBot(pendingQuestion);
            pendingQuestion = null;
          }
        } else {
          submitBtn.textContent = "Continue chatting \u2192";
          submitBtn.disabled = false;
          // The server does real email/phone validation now, so show what it
          // actually said rather than a blanket "something went wrong" the
          // visitor can't act on. Every detail on this endpoint is written
          // for an end user, and it's escaped on the way into the DOM.
          let detail = "";
          try { detail = (await res.json()).detail || ""; } catch (e) {}
          errorEl.textContent =
            typeof detail === "string" && detail
              ? detail
              : "Something went wrong. Please try again.";
          errorEl.style.display = "block";
        }
      } catch (e) {
        submitBtn.textContent = "Continue chatting \u2192";
        submitBtn.disabled = false;
        errorEl.textContent = "Network error. Please try again.";
        errorEl.style.display = "block";
      }
    };
  }

  // ---------------- ASK BOT ----------------
  async function askBot(question) {
    const typing = showTyping();
    // Disabled while waiting on a reply — previously a visitor could fire
    // off several messages before the first response landed, and since
    // requests can resolve out of order, replies could show up in a
    // different order than the questions were asked.
    blockInput("Waiting for reply...");

    // FIX: previously had no error handling at all — a network failure or
    // non-2xx response left the "..." typing indicator stuck forever with
    // no message and no way to retry, since res.json() would throw and the
    // rejection was never caught.
    try {
      const res = await fetch(`${apiBase}/public/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          projectId,
          message: question,
          sessionId,
          // Durable per-browser id. The server keys the lead-capture gate on
          // this, not on sessionId, so a visitor who already gave their
          // details isn't asked again when their 3-hour session rolls over.
          visitorId: userId,
          accessToken,
        }),
      });

      if (res.status === 401) {
        // Either this project just turned on a password, or our token
        // expired. Ask for it and retry the same question once unlocked.
        typing.remove();
        saveAccessToken(null);
        pendingQuestion = question;
        showPasswordForm();
        return;
      }

      if (res.status === 403) {
        // Refused: this website isn't on the project's Allowed websites list
        // (or the chat is switched off). The body is {detail}, not an answer,
        // so it used to fall through to a vague "something went wrong" that
        // gave a merchant who forgot to list their site no clue why.
        const refusal = await res.json().catch(() => ({}));
        typing.remove();
        console.warn("[Zavo chat] Request refused: " + (refusal.detail || "this website isn't allowed.") +
          " Add this website under Integrations > Embeddable Chat Widget > Allowed websites.");
        addMsg("assistant", render(refusal.detail || "This assistant isn't available here."));
        return;
      }

      const data = await res.json();
      typing.remove();
      if (data.sessionId) saveSessionId(data.sessionId);

      // The gate is enforced server-side now, so this can fire even when the
      // local counter hasn't tripped — a fresh browser resuming an older
      // conversation, or trigger_after_messages changed since page load.
      if (data.leadRequired) {
        if (data.leadForm) leadConfig = { ...(leadConfig || {}), ...data.leadForm };
        pendingQuestion = question;
        showLeadForm();
        return;
      }

      // A person is handling this chat: the message was delivered to them,
      // there may be no bot answer, and their replies arrive by polling.
      if (data.status === "human") {
        if (data.answer) addMsg("assistant", render(data.answer));
        startHumanPoll();
        return;
      }

      addMsg("assistant", render(data.answer || "Sorry, something went wrong. Please try again."));
    } catch (e) {
      typing.remove();
      addMsg("assistant", "Sorry, something went wrong. Please try again.");
    } finally {
      if (!awaitingLead) unblockInput();
    }
  }

  // ---------------- FORM SUBMIT ----------------
  form.onsubmit = async (e) => {
    e.preventDefault();

    const text = input.value.trim();
    if (!text || awaitingLead) return;

    input.value = "";
    addMsg("user", render(text));
    userMessageCount++;

    // While a website flow is running it decides what typed text means (an
    // answer, a question for the AI, ...). Returns false when it isn't
    // involved, and the message goes to the AI exactly as before.
    if (flowHost.plugin && flowHost.plugin.onUserText(text)) return;

    // The gate used to be decided here, entirely in the browser — which meant
    // it could be skipped, and which broke outright at trigger_after_messages
    // = 1: the form appeared before any message had been sent, so no chat
    // session existed yet and /public/leads had nothing to verify against.
    // askBot now always runs; the server answers with leadRequired instead of
    // an answer, before spending anything on OpenAI.
    await askBot(text);
  };

  // ---------------- TEAM REPLIES ----------------
  // While a person from the merchant's team is handling this chat, check for
  // their replies: every 5s with the panel open, 20s otherwise, for up to 30
  // minutes. Stops as soon as the chat is handed back to the bot.
  var humanPoll = { timer: null, cursor: null, started: 0 };
  function startHumanPoll() {
    if (humanPoll.timer || !sessionId) return;
    humanPoll.started = humanPoll.started || Date.now();
    humanPoll.cursor = humanPoll.cursor || new Date().toISOString();
    function tick() {
      humanPoll.timer = null;
      if (Date.now() - humanPoll.started > 30 * 60 * 1000) return;
      fetch(`${apiBase}/public/chat/poll`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ projectId: projectId, sessionId: sessionId, after: humanPoll.cursor }),
      })
        .then(function (res) { return res.ok ? res.json() : null; })
        .then(function (d) {
          if (!d) return;
          (d.messages || []).forEach(function (m) { addMsg("assistant", render(m.text)); });
          if (d.cursor) humanPoll.cursor = d.cursor;
          if (d.status === "human") {
            var open = document.visibilityState === "visible" && box.classList.contains("open");
            humanPoll.timer = setTimeout(tick, open ? 5000 : 20000);
          } else {
            humanPoll.started = 0;
          }
        })
        .catch(function () { humanPoll.timer = setTimeout(tick, 20000); });
    }
    humanPoll.timer = setTimeout(tick, 5000);
  }

  // ---------------- WEBSITE FLOWS (optional plugin) ----------------
  // When the merchant has an active website flow, widget-flows.js is loaded
  // and attaches itself through flowHost. Without one, nothing below does
  // anything beyond a single small GET: the widget behaves exactly as before.
  var restoredDone = false;
  var restoredWaiters = [];
  function markRestored() {
    restoredDone = true;
    restoredWaiters.splice(0).forEach(function (cb) { try { cb(); } catch (e) {} });
  }

  var flowHost = {
    plugin: null,
    config: null,
    api: {
      version: 1,
      root: root, host: host, dock: dock, msgs: msgs, input: input, sendBtn: sendBtn, form: form,
      apiBase: apiBase, projectId: projectId, userId: userId,
      addMsg: addMsg, render: render, esc: esc, showTyping: showTyping,
      blockInput: blockInput, unblockInput: unblockInput, scrollToEnd: scrollToEnd,
      askBot: askBot, setOpen: setOpen,
      isOpen: function () { return box.classList.contains("open"); },
      getSessionId: function () { return sessionId; },
      saveSessionId: saveSessionId,
      isAwaitingLead: function () { return awaitingLead; },
      userMessageCount: function () { return userMessageCount; },
      whenRestored: function (cb) { if (restoredDone) cb(); else restoredWaiters.push(cb); },
    },
  };

  function loadFlows() {
    fetch(`${apiBase}/public/flow-config/${projectId}`)
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (cfg) {
        if (!cfg || !cfg.active) return;
        flowHost.config = cfg;
        window.__zavoFlowHosts = window.__zavoFlowHosts || {};
        window.__zavoFlowHosts[projectId] = flowHost;
        var s = document.createElement("script");
        s.src = `${apiBase}/static/widget-flows.js?v=1`;
        s.async = true;
        s.setAttribute("data-project", projectId);
        document.head.appendChild(s);
      })
      .catch(function () {});
  }

  // ---------------- INIT ----------------
  fetchLeadConfig();
  fetchWidgetConfig();
  restoreHistory();
  loadFlows();
})();