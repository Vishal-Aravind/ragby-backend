(function () {
  const script = document.currentScript;
  if (!script) return;

  const projectId = script.dataset.project;
  if (!projectId) return;

  const apiBase = new URL(script.src).origin;
  const CHAT_TTL = 30 * 60 * 1000;

  let lead = JSON.parse(localStorage.getItem("chat_lead") || "null");
  let awaitingLead = false;
  let pendingQuestion = null;

  const storedChat = JSON.parse(localStorage.getItem("chat_session") || "null");
  const history =
    storedChat && Date.now() - storedChat.ts < CHAT_TTL
      ? storedChat.history
      : [];

  function persistChat() {
    localStorage.setItem(
      "chat_session",
      JSON.stringify({ ts: Date.now(), history })
    );
  }

  function render(text) {
    return (text || "")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\n/g, "<br/>");
  }

  // ---------------- CSS (typing dots) ----------------
  const style = document.createElement("style");
  style.innerHTML = `
    .typing {
      display:inline-flex;
      gap:4px;
      padding:8px 10px;
      background:#f4f4f5;
      border-radius:8px;
    }
    .typing span {
      width:6px;
      height:6px;
      background:#888;
      border-radius:50%;
      animation: blink 1.4s infinite both;
    }
    .typing span:nth-child(2){animation-delay:.2s}
    .typing span:nth-child(3){animation-delay:.4s}
    @keyframes blink {
      0%{opacity:.2}
      20%{opacity:1}
      100%{opacity:.2}
    }
  `;
  document.head.appendChild(style);

  // ---------------- UI ----------------
  const btn = document.createElement("div");
  btn.innerHTML = "💬";
  btn.style.cssText = `
    position:fixed;bottom:20px;right:20px;
    width:52px;height:52px;background:#000;color:#fff;
    border-radius:50%;display:flex;align-items:center;
    justify-content:center;cursor:pointer;z-index:999999;
  `;
  document.body.appendChild(btn);

  const box = document.createElement("div");
  box.style.cssText = `
    position:fixed;bottom:80px;right:20px;
    width:340px;height:460px;background:#fff;
    border-radius:12px;box-shadow:0 10px 30px rgba(0,0,0,.25);
    display:none;flex-direction:column;z-index:999999;
    font-family:system-ui;
  `;
  box.innerHTML = `
    <div style="padding:12px;border-bottom:1px solid #eee;font-weight:600">
      Ask us
    </div>
    <div id="msgs" style="flex:1;padding:12px;overflow:auto;font-size:14px"></div>
    <form id="chatForm" style="display:flex;border-top:1px solid #eee">
      <input id="input" placeholder="Type your question..."
        style="flex:1;padding:10px;border:none;outline:none"/>
      <button type="submit"
        style="padding:10px 14px;border:none;background:#000;color:#fff">
        Send
      </button>
    </form>
  `;
  document.body.appendChild(box);

  btn.onclick = () => {
    box.style.display = box.style.display === "none" ? "flex" : "none";
  };

  const msgs = box.querySelector("#msgs");
  const form = box.querySelector("#chatForm");
  const input = box.querySelector("#input");

  function addMsg(role, html) {
    const el = document.createElement("div");
    el.style.marginBottom = "10px";
    el.style.textAlign = role === "user" ? "right" : "left";
    el.innerHTML = `
      <div style="
        display:inline-block;
        background:${role === "user" ? "#000" : "#f4f4f5"};
        color:${role === "user" ? "#fff" : "#000"};
        padding:8px 10px;border-radius:8px;max-width:85%">
        ${html}
      </div>
    `;
    msgs.appendChild(el);
    msgs.scrollTop = msgs.scrollHeight;
    return el;
  }

  history.forEach((m) => addMsg(m.role, render(m.content)));

  function showTyping() {
    const el = document.createElement("div");
    el.style.marginBottom = "10px";
    el.innerHTML = `
      <div class="typing">
        <span></span><span></span><span></span>
      </div>
    `;
    msgs.appendChild(el);
    msgs.scrollTop = msgs.scrollHeight;
    return el;
  }

  async function askBot(question) {
    const typing = showTyping();

    const res = await fetch(`${apiBase}/public/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ projectId, message: question, history }),
    });

    const data = await res.json();
    typing.remove();

    addMsg("assistant", render(data.answer || ""));
    history.push({ role: "assistant", content: data.answer || "" });
    persistChat();
  }

  function showLeadForm() {
    awaitingLead = true;

    const id = "lead-form-" + Date.now();
    addMsg(
      "assistant",
      `
      <div id="${id}">
        <div style="margin-bottom:6px;font-weight:500">
          Before we continue, please share:
        </div>
        <input placeholder="Name" id="lf-name"
          style="width:100%;padding:6px;margin-bottom:6px"/>
        <input placeholder="Phone" id="lf-phone"
          style="width:100%;padding:6px;margin-bottom:6px"/>
        <input placeholder="Email (optional)" id="lf-email"
          style="width:100%;padding:6px;margin-bottom:8px"/>
        <button style="width:100%;padding:8px;background:#000;color:#fff;border:none">
          Submit
        </button>
      </div>
      `
    );

    const container = msgs.querySelector(`#${id}`);
    container.querySelector("button").onclick = async () => {
      const name = container.querySelector("#lf-name").value.trim();
      const phone = container.querySelector("#lf-phone").value.trim();
      const email = container.querySelector("#lf-email").value.trim();

      if (!name || !phone) return;

      lead = { name, phone, email };
      localStorage.setItem("chat_lead", JSON.stringify(lead));

      await fetch(`${apiBase}/public/lead`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ projectId, ...lead }),
      });

      awaitingLead = false;
      container.remove();
      addMsg("assistant", "Thanks!");

      if (pendingQuestion) {
        await askBot(pendingQuestion);
        pendingQuestion = null;
      }
    };
  }

  // ---------------- CHAT ----------------
  form.onsubmit = async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text || awaitingLead) return;
    input.value = "";

    addMsg("user", render(text));
    history.push({ role: "user", content: text });
    persistChat();

    const userCount = history.filter((h) => h.role === "user").length;

    if (!lead && userCount === 2) {
      pendingQuestion = text;
      showLeadForm();
      return;
    }

    await askBot(text);
  };
})();
