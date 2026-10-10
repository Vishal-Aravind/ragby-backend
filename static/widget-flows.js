/*
 * Zavo website flows - plugin for widget.js.
 *
 * Loaded by widget.js only when the project has an active website flow. It
 * attaches through window.__zavoFlowHosts[projectId] and renders what the
 * flow engine (backend/web_flows) returns: chips, buttons, lists,
 * carousels, forms, ratings, media and links, plus auto-open triggers.
 *
 * Rules: every piece of DOM is built with createElement/textContent (only
 * plain text bubbles go through widget.js's own escape+linkify). Links are
 * scheme-checked here as well as on the server. Source stays ASCII-only.
 */
(function () {
  var me = document.currentScript;
  var pid = me && me.getAttribute("data-project");
  var hosts = window.__zavoFlowHosts || {};
  var host = pid && hosts[pid];
  if (!host || host.plugin || !host.api || host.api.version !== 1) return;

  var api = host.api;
  var cfg = host.config || {};
  var root = api.root;
  var msgs = api.msgs;

  var FLOW_KEY = "zavo_flow_" + pid;        // which chat session is in a flow
  var PT_SESSION_KEY = "zavo_pt_s_" + pid;  // proactive: fired this browser session
  var PT_KEY = "zavo_pt_" + pid;            // proactive: {lastFiredAt, closedAt}

  var state = {
    sessionId: null, seq: 0, nodeId: null, status: null,
    busy: false, liveInput: null, continueTimer: null,
    pollTimer: null, pollCursor: null, pollStarted: 0,
    disabled: false, proactiveFired: false,
  };

  function lsGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
  function lsSet(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
  function lsDel(k) { try { localStorage.removeItem(k); } catch (e) {} }
  function ssGet(k) { try { return sessionStorage.getItem(k); } catch (e) { return null; } }
  function ssSet(k, v) { try { sessionStorage.setItem(k, v); } catch (e) {} }

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function safeUrl(url, schemes) {
    if (typeof url !== "string") return null;
    var m = /^([a-z][a-z0-9+.-]*):/i.exec(url.trim());
    if (!m) return null;
    return (schemes || ["http", "https"]).indexOf(m[1].toLowerCase()) >= 0 ? url.trim() : null;
  }
  function finePointer() {
    return !!(window.matchMedia && window.matchMedia("(hover: hover) and (pointer: fine)").matches);
  }
  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }

  // ------------------------------------------------------------------ styles
  var css = el("style");
  css.textContent = [
    ".zf-row { display: flex; margin: -2px 0 12px; }",
    ".zf-opts { display: flex; flex-wrap: wrap; gap: 8px; max-width: 100%; }",
    ".zf-opts.stack { flex-direction: column; align-items: stretch; width: 84%; }",
    ".zf-chip { border: 1.5px solid var(--c2); background: #fff; color: var(--c1); border-radius: 999px; padding: 8px 14px;",
    "  font-size: 13.5px; font-weight: 600; cursor: pointer; line-height: 1.2; text-align: center;",
    "  transition: background .15s, color .15s, transform .15s; }",
    ".zf-opts.stack .zf-chip { border-radius: 12px; }",
    ".zf-chip:hover:not(:disabled) { background: var(--c2); color: #fff; transform: translateY(-1px); }",
    ".zf-chip:focus-visible { outline: 3px solid var(--ring); outline-offset: 2px; }",
    ".zf-chip:disabled { opacity: .45; cursor: default; }",
    ".zf-chip[aria-pressed=true] { background: var(--c2); color: #fff; opacity: 1; }",
    ".zf-filter { width: 100%; margin-bottom: 8px; }",
    ".zf-more { background: none; border: none; color: var(--c1); font-weight: 600; font-size: 13px; cursor: pointer; padding: 4px 2px; }",
    ".zf-car { display: flex; gap: 10px; overflow-x: auto; scroll-snap-type: x mandatory; padding: 2px 2px 8px; width: 100%; }",
    ".zf-car:focus-visible { outline: 3px solid var(--ring); }",
    ".zf-card { flex: 0 0 78%; scroll-snap-align: start; background: #fff; border: 1px solid #ebe9f7; border-radius: 16px; overflow: hidden;",
    "  box-shadow: 0 2px 8px rgba(79,70,229,.07); display: flex; flex-direction: column; }",
    ".zf-card img { width: 100%; height: 130px; object-fit: cover; display: block; background: #f3f0ff; }",
    ".zf-card-body { padding: 10px 12px; flex: 1; }",
    ".zf-card-title { font-weight: 650; font-size: 14px; }",
    ".zf-card-text { font-size: 12.5px; color: #6b7280; margin-top: 3px; }",
    ".zf-card-btns { display: flex; flex-direction: column; gap: 6px; padding: 0 12px 12px; }",
    ".zf-car-nav { display: flex; gap: 6px; justify-content: flex-end; width: 100%; }",
    ".zf-car-nav button { width: 30px; height: 30px; border-radius: 50%; border: 1px solid #e5e3f3; background: #fff; cursor: pointer; }",
    ".zf-link { display: inline-block; margin-top: 8px; padding: 8px 14px; border-radius: 12px; background: var(--grad); color: #fff !important;",
    "  font-weight: 600; font-size: 13.5px; text-decoration: none !important; }",
    ".zf-media { max-width: 100%; border-radius: 12px; display: block; }",
    ".zf-cap { margin-top: 6px; }",
    ".zf-file { display: flex; gap: 8px; align-items: center; color: inherit !important; text-decoration: none !important; font-weight: 600; }",
    ".zf-stars { display: flex; gap: 4px; flex-wrap: wrap; }",
    ".zf-star { border: none; background: none; font-size: 26px; line-height: 1; cursor: pointer; color: #d1d5db; padding: 2px; }",
    ".zf-star.on, .zf-star:hover { color: #f59e0b; }",
    ".zf-nps { min-width: 34px; padding: 7px 0; }",
    ".zf-check { display: flex; gap: 8px; align-items: flex-start; font-size: 13px; margin-bottom: 8px; }",
    ".zf-label { display: block; font-size: 12.5px; font-weight: 600; margin: 0 0 4px; color: #374151; }",
    ".zf-teaser { position: fixed; right: 20px; bottom: 92px; max-width: 260px; background: #fff; color: #1f2937; border-radius: 16px;",
    "  padding: 12px 34px 12px 14px; font: 14px/1.4 -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;",
    "  box-shadow: 0 12px 32px rgba(49,46,129,.25); z-index: 2147483000; cursor: pointer; animation: zv-in .4s both; }",
    ".zf-teaser-x { position: absolute; top: 6px; right: 6px; width: 24px; height: 24px; border: none; border-radius: 50%;",
    "  background: #f3f4f6; cursor: pointer; font-size: 14px; line-height: 1; }",
    "@media (max-width: 480px) { .fld, .zf-filter { font-size: 16px; } .zf-teaser { right: 14px; bottom: 86px; } }",
  ].join("\n");
  root.appendChild(css);

  // --------------------------------------------------------------- network
  function call(path, body) {
    return fetch(api.apiBase + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(function (res) {
      if (res.status === 403 || res.status === 404) {
        disable();
        return null;
      }
      if (res.status === 429) return { error: { code: "rate_limited", message: "Please wait a moment and try again." } };
      return res.ok ? res.json() : null;
    }).catch(function () { return null; });
  }

  function page() {
    return { url: String(location.href).slice(0, 1000), path: String(location.pathname).slice(0, 500),
             title: String(document.title || "").slice(0, 300) };
  }

  function disable() {
    // Refused (site not allowed / project gone): get out of the way and let
    // the widget work exactly as it does without flows.
    state.disabled = true;
    clearTimers();
    lsDel(FLOW_KEY);
  }

  function clearTimers() {
    if (state.continueTimer) clearTimeout(state.continueTimer);
    if (state.pollTimer) clearTimeout(state.pollTimer);
    state.continueTimer = state.pollTimer = null;
  }

  // ------------------------------------------------------------- lifecycle
  function startFlow(via) {
    if (state.disabled || state.busy) return Promise.resolve();
    state.busy = true;
    var typing = api.showTyping();
    return call("/public/flow/start", {
      projectId: pid, sessionId: api.getSessionId() || null, visitorId: api.userId,
      via: via || "open", page: page(),
    }).then(function (env) {
      typing.remove();
      state.busy = false;
      if (!env || env.status === "inactive") { disable(); return; }
      return apply(env);
    });
  }

  function step(action, nodeId) {
    if (state.disabled || !state.sessionId) return Promise.resolve();
    if (state.busy) return Promise.resolve();
    state.busy = true;
    var typing = api.showTyping();
    return call("/public/flow/step", {
      projectId: pid, sessionId: state.sessionId, visitorId: api.userId,
      seq: state.seq, nodeId: nodeId || null, action: action, page: page(),
    }).then(function (env) {
      typing.remove();
      state.busy = false;
      if (!env) {
        api.addMsg("assistant", "Sorry, something went wrong. Please try again.");
        reenableLive();
        return;
      }
      return apply(env, action);
    });
  }

  function apply(env, action) {
    if (env.error && env.error.code === "rate_limited") {
      api.addMsg("assistant", api.esc(env.error.message));
      reenableLive();
      return;
    }
    if (env.status === "inactive") {
      disable();
      if (action && action.type === "text") return api.askBot(action.text);
      return;
    }
    if (env.status === "expired") {
      lsDel(FLOW_KEY);
      state.sessionId = null;
      api.addMsg("assistant", api.esc("Let's start again."));
      return startFlow("restart");
    }
    if (env.sessionId) {
      state.sessionId = env.sessionId;
      api.saveSessionId(env.sessionId);  // also restarts the 7-day memory clock
      lsSet(FLOW_KEY, env.sessionId);
    }
    var prevNode = state.nodeId;
    state.seq = env.seq;
    state.nodeId = env.nodeId;
    state.status = env.status;

    if (env.resync) {
      renderInput(env);
      return;
    }
    return renderMessages(env.replay ? replayToMessages(env.replay) : [])
      .then(function () { return renderMessages(env.messages || []); })
      .then(function () {
        var invalid = env.error && env.error.code === "invalid";
        if (invalid && state.liveInput && env.nodeId === prevNode) {
          // Same question, rejected answer: keep what they typed, mark it.
          showError(env.error, state.liveInput);
          reenableLive();
          return;
        }
        if (invalid) showError(env.error, null);
        if (env.delegate === "ai" && action && action.type === "text") {
          return api.askBot(action.text).then(function () { renderInput(env); });
        }
        renderInput(env);
        if (env.continueAfterMs != null) scheduleContinue(env.continueAfterMs);
        if (env.status === "human") startPolling();
      });
  }

  function replayToMessages(entries) {
    return (entries || []).map(function (e) { return { kind: "text", text: e.content, role: e.role }; });
  }

  function scheduleContinue(ms) {
    if (state.continueTimer) clearTimeout(state.continueTimer);
    var nodeId = state.nodeId;
    state.continueTimer = setTimeout(function () {
      state.continueTimer = null;
      step({ type: "continue" }, nodeId);
    }, Math.max(0, ms));
  }

  // -------------------------------------------------------------- rendering
  function bubbleRow(role) {
    var row = el("div", "row " + (role === "user" ? "user" : "assistant"));
    var bubble = el("div", "bubble");
    row.appendChild(bubble);
    msgs.appendChild(row);
    return bubble;
  }

  function renderMessages(list) {
    var p = Promise.resolve();
    list.forEach(function (m) {
      p = p.then(function () { return renderMessage(m); });
    });
    return p;
  }

  function renderMessage(m) {
    if (!m || !m.kind) return;
    if (m.kind === "pause") {
      var t = api.showTyping();
      return sleep(Math.min(m.ms || 0, 10000)).then(function () { t.remove(); });
    }
    if (m.kind === "text") {
      api.addMsg(m.role === "user" ? "user" : "assistant", api.render(m.text || ""));
      return;
    }
    var b = bubbleRow("assistant");
    if (m.kind === "image") {
      var url = safeUrl(m.url, ["https"]);
      if (url) {
        var img = el("img", "zf-media");
        img.alt = m.caption || "";
        img.loading = "lazy";
        img.onerror = function () { img.replaceWith(linkEl(url, "Open image")); };
        img.onload = function () { api.scrollToEnd(true); };
        img.src = url;
        b.appendChild(img);
      }
    } else if (m.kind === "video" || m.kind === "audio") {
      var murl = safeUrl(m.url, ["https"]);
      if (murl) {
        var media = el(m.kind, "zf-media");
        media.controls = true;
        media.preload = m.kind === "video" ? "metadata" : "none";
        if (m.kind === "video") media.setAttribute("playsinline", "");
        media.src = murl;
        b.appendChild(media);
      }
    } else if (m.kind === "file") {
      var furl = safeUrl(m.url, ["https"]);
      if (furl) {
        var a = el("a", "zf-file", String.fromCharCode(0xD83D, 0xDCC4) + " " + (m.name || "Document"));
        a.href = furl; a.target = "_blank"; a.rel = "noopener noreferrer";
        b.appendChild(a);
      }
    } else if (m.kind === "link") {
      if (m.text) b.appendChild(el("div", null, m.text));
      var lurl = safeUrl(m.url, ["http", "https", "tel", "mailto"]);
      if (lurl) b.appendChild(linkEl(lurl, m.label || "Open"));
      return;
    } else if (m.kind === "card") {
      b.className += " card";
      var curl = safeUrl(m.image, ["https"]);
      if (curl) { var ci = el("img", "zf-media"); ci.alt = ""; ci.src = curl; b.appendChild(ci); }
      if (m.title) b.appendChild(el("div", "card-title", m.title));
      if (m.text) b.appendChild(el("div", "zf-cap", m.text));
      (m.links || []).forEach(function (l) {
        var u = safeUrl(l.url, ["http", "https", "tel", "mailto"]);
        if (u) { b.appendChild(el("br")); b.appendChild(linkEl(u, l.label)); }
      });
      return;
    }
    if (m.caption) b.appendChild(el("div", "zf-cap", m.caption));
    api.scrollToEnd(true);
  }

  function linkEl(url, label) {
    var a = el("a", "zf-link", label || "Open");
    a.href = url;
    if (!/^(tel|mailto):/i.test(url)) { a.target = "_blank"; a.rel = "noopener noreferrer"; }
    return a;
  }

  function retireLive() {
    if (state.liveInput) {
      state.liveInput.querySelectorAll("button, input, select").forEach(function (x) { x.disabled = true; });
      if (state.liveInput.getAttribute("data-temp") === "1") state.liveInput.remove();
      state.liveInput = null;
    }
    resetComposer();
  }

  function reenableLive() {
    if (state.liveInput) {
      state.liveInput.querySelectorAll("button, input, select").forEach(function (x) { x.disabled = false; });
      state.liveInput.querySelectorAll("[aria-pressed=true]").forEach(function (x) { x.setAttribute("aria-pressed", "false"); });
    }
  }

  function resetComposer() {
    api.input.setAttribute("placeholder", "Type your question...");
    api.input.removeAttribute("inputmode");
    api.input.setAttribute("autocomplete", "off");
  }

  function renderInput(env) {
    retireLive();
    var inp = env.input;
    if (env.menuChip && env.status === "ai") {
      inp = { kind: "choices", layout: "chips", options: [{ id: "__menu", label: "Back to menu" }], menu: true };
    }
    if (!inp) return;
    var nodeId = env.nodeId;
    if (inp.kind === "choices") return renderChoices(inp, nodeId);
    if (inp.kind === "carousel") return renderCarousel(inp, nodeId);
    if (inp.kind === "form") return renderForm(inp, nodeId);
    if (inp.kind === "rating") return renderRating(inp, nodeId);
    if (inp.kind === "field") return renderField(inp, nodeId);
  }

  function inputRow(temp) {
    var row = el("div", "zf-row");
    if (temp) row.setAttribute("data-temp", "1");
    msgs.appendChild(row);
    state.liveInput = row;
    return row;
  }

  function choose(btn, group, label, action, nodeId) {
    if (state.busy) return;
    group.querySelectorAll("button").forEach(function (x) { x.disabled = true; });
    btn.setAttribute("aria-pressed", "true");
    api.addMsg("user", api.render(label));
    step(action, nodeId);
  }

  function renderChoices(inp, nodeId) {
    var row = inputRow(!!inp.menu);
    var stack = inp.layout === "buttons" || inp.layout === "list";
    var group = el("div", "zf-opts" + (stack ? " stack" : ""));
    group.setAttribute("role", "group");
    group.setAttribute("aria-label", "Options");
    var options = inp.options || [];
    var buttons = options.map(function (o) {
      var b = el("button", "zf-chip", o.label);
      b.type = "button";
      b.setAttribute("aria-pressed", "false");
      b.onclick = function () {
        if (o.id === "__menu") choose(b, group, o.label, { type: "menu" }, null);
        else choose(b, group, o.label, { type: "choice", id: o.id }, nodeId);
      };
      return b;
    });
    var LIMIT = 6;
    if (inp.layout === "list" && options.length > 12) {
      var filter = el("input", "fld zf-filter");
      filter.placeholder = "Search options...";
      filter.setAttribute("aria-label", "Search options");
      filter.oninput = function () {
        var q = filter.value.trim().toLowerCase();
        buttons.forEach(function (b) { b.style.display = !q || b.textContent.toLowerCase().indexOf(q) >= 0 ? "" : "none"; });
      };
      row.appendChild(filter);
    }
    buttons.forEach(function (b, i) {
      if (inp.layout === "list" && i >= LIMIT) b.style.display = "none";
      group.appendChild(b);
    });
    var wrap = el("div");
    wrap.style.width = "100%";
    if (inp.layout === "list" && options.length > 12) wrap.appendChild(row.firstChild);
    wrap.appendChild(group);
    if (inp.layout === "list" && options.length > LIMIT) {
      var more = el("button", "zf-more", "Show all " + options.length + " options");
      more.type = "button";
      more.onclick = function () { buttons.forEach(function (b) { b.style.display = ""; }); more.remove(); };
      wrap.appendChild(more);
    }
    row.replaceChildren(wrap);
    api.scrollToEnd(true);
  }

  function renderCarousel(inp, nodeId) {
    var row = inputRow(false);
    var wrap = el("div");
    wrap.style.width = "100%";
    var track = el("div", "zf-car");
    track.tabIndex = 0;
    track.setAttribute("role", "region");
    track.setAttribute("aria-roledescription", "carousel");
    track.setAttribute("aria-label", "Options");
    var cards = inp.cards || [];
    cards.forEach(function (c, i) {
      var card = el("div", "zf-card");
      card.setAttribute("role", "group");
      card.setAttribute("aria-label", (i + 1) + " of " + cards.length);
      var img = safeUrl(c.image, ["https"]);
      if (img) { var im = el("img"); im.alt = ""; im.loading = "lazy"; im.src = img; card.appendChild(im); }
      var body = el("div", "zf-card-body");
      if (c.title) body.appendChild(el("div", "zf-card-title", c.title));
      if (c.text) body.appendChild(el("div", "zf-card-text", c.text));
      card.appendChild(body);
      var btns = el("div", "zf-card-btns");
      (c.buttons || []).forEach(function (bt) {
        if (bt.url) {
          var u = safeUrl(bt.url, ["http", "https", "tel", "mailto"]);
          if (u) { var a = linkEl(u, bt.label); a.style.marginTop = "0"; a.style.textAlign = "center"; btns.appendChild(a); }
        } else {
          var b = el("button", "zf-chip", bt.label);
          b.type = "button";
          b.onclick = function () {
            choose(b, track, (c.title ? c.title + " - " : "") + bt.label, { type: "choice", id: bt.id }, nodeId);
          };
          btns.appendChild(b);
        }
      });
      card.appendChild(btns);
      track.appendChild(card);
    });
    track.addEventListener("keydown", function (e) {
      if (e.key === "ArrowRight") { track.scrollBy({ left: track.clientWidth * 0.8, behavior: "smooth" }); e.preventDefault(); }
      if (e.key === "ArrowLeft") { track.scrollBy({ left: -track.clientWidth * 0.8, behavior: "smooth" }); e.preventDefault(); }
    });
    wrap.appendChild(track);
    if (cards.length > 1 && finePointer()) {
      var nav = el("div", "zf-car-nav");
      var prev = el("button", null, String.fromCharCode(0x2039)); prev.type = "button"; prev.setAttribute("aria-label", "Previous");
      var next = el("button", null, String.fromCharCode(0x203A)); next.type = "button"; next.setAttribute("aria-label", "Next");
      prev.onclick = function () { track.scrollBy({ left: -track.clientWidth * 0.8, behavior: "smooth" }); };
      next.onclick = function () { track.scrollBy({ left: track.clientWidth * 0.8, behavior: "smooth" }); };
      nav.appendChild(prev); nav.appendChild(next);
      wrap.appendChild(nav);
    }
    row.appendChild(wrap);
    api.scrollToEnd(true);
  }

  var fieldSeq = 0;
  function renderForm(inp, nodeId) {
    var row = inputRow(false);
    row.className = "row assistant";
    var card = el("div", "bubble card");
    var formEl = el("form");
    formEl.noValidate = true;
    if (inp.title) card.appendChild(el("div", "card-title", inp.title));
    var summary = el("div", "err");
    summary.setAttribute("role", "alert");
    summary.style.display = "none";
    var controls = {};
    (inp.fields || []).forEach(function (f) {
      var id = "zf-f" + (++fieldSeq);
      var errId = id + "-e";
      var wrap = el("div");
      var ctrl;
      if (f.type === "consent") {
        var lab = el("label", "zf-check");
        ctrl = el("input");
        ctrl.type = "checkbox";
        ctrl.id = id;
        lab.appendChild(ctrl);
        lab.appendChild(el("span", null, f.label + (f.required ? " *" : "")));
        wrap.appendChild(lab);
      } else {
        var label = el("label", "zf-label", f.label + (f.required ? " *" : ""));
        label.htmlFor = id;
        wrap.appendChild(label);
        if (f.type === "choice") {
          ctrl = el("select", "fld");
          var blank = el("option", null, "Choose...");
          blank.value = "";
          ctrl.appendChild(blank);
          (f.options || []).forEach(function (o) { var op = el("option", null, o); op.value = o; ctrl.appendChild(op); });
        } else {
          ctrl = el("input", "fld");
          ctrl.type = { email: "email", phone: "tel", number: "text", date: "date" }[f.type] || "text";
          if (f.type === "number") ctrl.inputMode = "decimal";
          if (f.type === "phone") ctrl.inputMode = "tel";
          ctrl.autocomplete = { name: "name", email: "email", phone: "tel" }[f.name] || "off";
          if (f.placeholder) ctrl.placeholder = f.placeholder;
          ctrl.maxLength = 1000;
        }
        ctrl.id = id;
        wrap.appendChild(ctrl);
      }
      ctrl.setAttribute("aria-describedby", errId);
      var err = el("div", "err");
      err.id = errId;
      err.style.display = "none";
      wrap.appendChild(err);
      controls[f.name] = { ctrl: ctrl, err: err, field: f };
      formEl.appendChild(wrap);
      ctrl.addEventListener("focus", function () {
        if (!finePointer()) setTimeout(function () { ctrl.scrollIntoView({ block: "center" }); }, 300);
      });
    });
    var submit = el("button", "cta", inp.submitLabel || "Send");
    submit.type = "submit";
    formEl.appendChild(summary);
    formEl.appendChild(submit);
    formEl.onsubmit = function (e) {
      e.preventDefault();
      if (state.busy) return;
      var values = {}, errors = {};
      Object.keys(controls).forEach(function (name) {
        var c = controls[name], f = c.field;
        var v = f.type === "consent" ? c.ctrl.checked : c.ctrl.value.trim();
        values[name] = v;
        if (f.required && (v === "" || v === false)) errors[name] = f.type === "consent" ? "Please tick this box to continue." : "This is required.";
        else if (f.type === "email" && v && !/^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$/.test(v)) errors[name] = "Please enter a valid email address.";
        else if (f.type === "phone" && v && (v.replace(/\D/g, "").length < 7 || v.replace(/\D/g, "").length > 15)) errors[name] = "Please enter a valid phone number.";
      });
      if (markErrors(controls, summary, errors)) return;
      submit.disabled = true;
      step({ type: "form", values: values }, nodeId);
    };
    row._zfForm = { controls: controls, summary: summary };
    card.appendChild(formEl);
    row.appendChild(card);
    api.scrollToEnd(true);
  }

  function markErrors(controls, summary, errors) {
    var names = Object.keys(errors || {});
    Object.keys(controls).forEach(function (name) {
      var c = controls[name];
      var msg = errors[name];
      c.ctrl.setAttribute("aria-invalid", msg ? "true" : "false");
      c.err.textContent = msg || "";
      c.err.style.display = msg ? "block" : "none";
    });
    summary.textContent = names.length ? "Please check the highlighted fields." : "";
    summary.style.display = names.length ? "block" : "none";
    if (names.length && controls[names[0]]) controls[names[0]].ctrl.focus();
    return names.length > 0;
  }

  function showError(error, live) {
    var fc = live && live._zfForm;
    if (fc && error.fields) {
      // Server-side validation of a form (the same rules as above, but
      // authoritative): map the messages back onto the fields.
      var mapped = markErrors(fc.controls, fc.summary, error.fields);
      if (mapped) return;
    }
    api.addMsg("assistant", api.esc(error.message || "Please check your answer."));
  }

  function renderRating(inp, nodeId) {
    var row = inputRow(false);
    var group = el("div", inp.style === "nps" ? "zf-opts" : "zf-stars");
    group.setAttribute("role", "group");
    group.setAttribute("aria-label", "Rating");
    var max = inp.style === "nps" ? 10 : 5;
    for (var i = inp.style === "nps" ? 0 : 1; i <= max; i++) {
      (function (v) {
        var b = inp.style === "nps" ? el("button", "zf-chip zf-nps", String(v)) : el("button", "zf-star", String.fromCharCode(0x2605));
        b.type = "button";
        b.setAttribute("aria-label", "Rate " + v + " of " + max);
        if (inp.style !== "nps") {
          b.onmouseenter = function () { group.querySelectorAll(".zf-star").forEach(function (s, k) { s.classList.toggle("on", k < v); }); };
        }
        b.onclick = function () { choose(b, group, "Rated " + v + "/" + max, { type: "rating", value: v }, nodeId); };
        group.appendChild(b);
      })(i);
    }
    row.appendChild(group);
    api.scrollToEnd(true);
  }

  function renderField(inp, nodeId) {
    if (inp.type === "date") {
      // Dates get a picker card; everything else is typed in the composer.
      var row = inputRow(false);
      row.className = "row assistant";
      var card = el("div", "bubble card");
      var f = el("form");
      var d = el("input", "fld");
      d.type = "date";
      d.setAttribute("aria-label", "Pick a date");
      if (inp.min && /^\d{4}-\d{2}-\d{2}$/.test(inp.min)) d.min = inp.min;
      if (inp.max && /^\d{4}-\d{2}-\d{2}$/.test(inp.max)) d.max = inp.max;
      var go = el("button", "cta", "Send");
      go.type = "submit";
      f.appendChild(d); f.appendChild(go);
      f.onsubmit = function (e) {
        e.preventDefault();
        if (!d.value || state.busy) return;
        go.disabled = true;
        api.addMsg("user", api.render(d.value));
        step({ type: "field", value: d.value }, nodeId);
      };
      card.appendChild(f);
      row.appendChild(card);
      api.scrollToEnd(true);
      return;
    }
    var ph = { email: "Type your email...", phone: "Type your phone number...", number: "Type a number..." }[inp.type];
    api.input.setAttribute("placeholder", inp.placeholder || ph || "Type your answer...");
    if (inp.type === "email") { api.input.setAttribute("inputmode", "email"); api.input.setAttribute("autocomplete", "email"); }
    if (inp.type === "phone") { api.input.setAttribute("inputmode", "tel"); api.input.setAttribute("autocomplete", "tel"); }
    if (inp.type === "number") api.input.setAttribute("inputmode", "decimal");
    if (api.isOpen() && finePointer()) api.input.focus();
  }

  // ------------------------------------------------------------ human mode
  function startPolling() {
    if (state.pollTimer || state.disabled) return;
    // One poller for the whole widget (it also notices takeovers): newer
    // widget.js does it, so replies are never shown twice.
    if (api.startHumanPoll) { api.startHumanPoll(); return; }
    state.pollStarted = state.pollStarted || Date.now();
    var tick = function () {
      state.pollTimer = null;
      if (state.disabled || state.status !== "human") return;
      if (Date.now() - state.pollStarted > 30 * 60 * 1000) return;
      call("/public/flow/poll", { projectId: pid, sessionId: state.sessionId, visitorId: api.userId, after: state.pollCursor })
        .then(function (res) {
          if (res && res.messages) {
            res.messages.forEach(function (m) { api.addMsg("assistant", api.render(m.text)); });
            if (res.cursor) state.pollCursor = res.cursor;
            if (res.status && res.status !== "human") state.status = res.status;
          }
          if (state.status === "human") {
            var visible = document.visibilityState === "visible" && api.isOpen();
            state.pollTimer = setTimeout(tick, visible ? 5000 : 20000);
          }
        });
    };
    state.pollCursor = state.pollCursor || new Date().toISOString();
    state.pollTimer = setTimeout(tick, 5000);
  }

  // ------------------------------------------------------- widget.js hooks
  function hasConversation() {
    return msgs.querySelector(".row.user") != null;
  }

  var plugin = {
    onOpen: function () {
      if (state.disabled || state.sessionId || cfg.startOnOpen === false) return false;
      if (hasConversation() || msgs.children.length) return false;
      startFlow(state.proactiveVia || "open");
      state.proactiveVia = null;
      return true;
    },
    onClose: function () {
      if (state.proactiveFired) {
        var pt = readPt();
        pt.closedAt = Date.now();
        lsSet(PT_KEY, JSON.stringify(pt));
      }
    },
    onUserText: function (text) {
      if (state.disabled || api.isAwaitingLead()) return false;
      if (state.sessionId) {
        if (state.busy) return true;
        retireLive();
        step({ type: "text", text: text }, state.nodeId);
        return true;
      }
      // No flow yet. A visitor with an earlier AI conversation keeps it;
      // a fresh visitor starts the flow (their first message opens it, as
      // on WhatsApp).
      if (msgs.querySelectorAll(".row.user").length > 1) return false;
      startFlow("text");
      return true;
    },
  };
  host.plugin = plugin;

  // Resume after a reload, or start if the panel was opened before we loaded.
  api.whenRestored(function () {
    var marker = lsGet(FLOW_KEY);
    var sid = api.getSessionId();
    if (marker && sid && marker === sid) {
      state.sessionId = sid;
      call("/public/flow/resume", { projectId: pid, sessionId: sid, visitorId: api.userId }).then(function (env) {
        if (!env) return;
        if (env.status === "expired" || env.status === "inactive") {
          lsDel(FLOW_KEY);
          state.sessionId = null;
          return;
        }
        apply(env);
      });
    } else if (api.isOpen() && !hasConversation() && cfg.startOnOpen !== false) {
      // Opened before this script arrived: replace the generic greeting.
      msgs.replaceChildren();
      startFlow("open");
    }
  });

  // ----------------------------------------------------- proactive triggers
  function readPt() {
    try { return JSON.parse(lsGet(PT_KEY) || "{}") || {}; } catch (e) { return {}; }
  }

  function canFire() {
    if (state.disabled || cfg.allowedHere === false || state.proactiveFired) return false;
    if (ssGet(PT_SESSION_KEY)) return false;
    if (api.isOpen() || hasConversation() || api.isAwaitingLead() || state.sessionId) return false;
    if (document.visibilityState !== "visible") return false;
    var pt = readPt(), now = Date.now();
    if (pt.lastFiredAt && now - pt.lastFiredAt < (cfg.cooldownHours || 0) * 3600000) return false;
    if (pt.closedAt && now - pt.closedAt < (cfg.suppressDays || 0) * 86400000) return false;
    return true;
  }

  function fire(type) {
    if (!canFire()) return;
    state.proactiveFired = true;
    ssSet(PT_SESSION_KEY, "1");
    var pt = readPt();
    pt.lastFiredAt = Date.now();
    lsSet(PT_KEY, JSON.stringify(pt));
    var mobile = !finePointer() || window.innerWidth <= 480;
    if (cfg.display === "teaser" || mobile) showTeaser(type);
    else {
      state.proactiveVia = "proactive:" + type;
      api.setOpen(true);
    }
  }

  function showTeaser(type) {
    var t = el("div", "zf-teaser");
    t.setAttribute("role", "status");
    t.appendChild(el("span", null, cfg.teaser || "Hi! Need any help?"));
    var x = el("button", "zf-teaser-x", String.fromCharCode(0x00D7));
    x.type = "button";
    x.setAttribute("aria-label", "Dismiss");
    x.onclick = function (e) {
      e.stopPropagation();
      t.remove();
      var pt = readPt();
      pt.closedAt = Date.now();
      lsSet(PT_KEY, JSON.stringify(pt));
    };
    t.appendChild(x);
    t.onclick = function () {
      t.remove();
      state.proactiveVia = "proactive:" + type;
      api.setOpen(true);
    };
    root.appendChild(t);
    var hide = setTimeout(function () { t.remove(); }, 30000);
    api.dock.addEventListener("click", function () { clearTimeout(hide); t.remove(); }, { once: true });
  }

  function urlMatches(rule) {
    var href = String(location.href).toLowerCase();
    var v = String(rule.value || "").toLowerCase();
    if (!v) return true;
    if (rule.match === "equals") return href === v || href.replace(/\/$/, "") === v.replace(/\/$/, "");
    if (rule.match === "starts_with") return href.indexOf(v) === 0;
    return href.indexOf(v) >= 0;
  }

  (cfg.triggers || []).forEach(function (rule) {
    if (rule.type === "time_on_page") {
      var visibleSecs = 0;
      var iv = setInterval(function () {
        if (state.proactiveFired || ssGet(PT_SESSION_KEY)) { clearInterval(iv); return; }
        if (document.visibilityState === "visible") visibleSecs++;
        if (visibleSecs >= (rule.seconds || 10)) { clearInterval(iv); fire("time"); }
      }, 1000);
    } else if (rule.type === "url_match") {
      var ivu = setInterval(function () {
        if (state.proactiveFired || ssGet(PT_SESSION_KEY)) { clearInterval(ivu); return; }
        if (urlMatches(rule)) { clearInterval(ivu); fire("url"); }
      }, 1000);
    } else if (rule.type === "exit_intent") {
      if (!finePointer()) return;
      var armedAt = Date.now() + 5000;
      document.addEventListener("mouseout", function (e) {
        if (Date.now() < armedAt || e.relatedTarget || e.clientY > 0) return;
        fire("exit");
      });
    } else if (rule.type === "scroll_depth") {
      var ticking = false;
      window.addEventListener("scroll", function () {
        if (ticking) return;
        ticking = true;
        requestAnimationFrame(function () {
          ticking = false;
          var doc = document.documentElement;
          var h = Math.max(doc.scrollHeight, document.body ? document.body.scrollHeight : 0);
          if (h < window.innerHeight * 1.2) return;
          var pct = (window.scrollY + window.innerHeight) / h * 100;
          if (pct >= (rule.percent || 50)) fire("scroll");
        });
      }, { passive: true });
    }
  });
})();
