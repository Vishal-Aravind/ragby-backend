(function () {
  const script = document.currentScript;
  if (!script) return;

  const projectId = script.dataset.project;
  if (!projectId) return;

  const apiBase = new URL(script.src).origin;

  // Session-based userId
  let userId = localStorage.getItem("rag_user_id");
  if (!userId) {
    userId = crypto.randomUUID();
    localStorage.setItem("rag_user_id", userId);
  }

  // Lead state
  let lead = JSON.parse(localStorage.getItem(`chat_lead_${projectId}`) || "null");
  let awaitingLead = false;
  let pendingQuestion = null;
  let userMessageCount = 0;

  // Lead capture config (fetched from backend)
  let leadConfig = null;

  const history = [];

  function render(text) {
    return (text || "")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\n/g, "<br/>");
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
    font-family:system-ui;overflow:hidden;
  `;

  box.innerHTML = `
    <div style="padding:12px;border-bottom:1px solid #eee;font-weight:600">
      Ask us
    </div>
    <div id="msgs" style="flex:1;padding:12px;overflow:auto;font-size:14px;position:relative"></div>
    <form id="chatForm" style="display:flex;border-top:1px solid #eee">
      <input id="chat-input" placeholder="Type your question..."
        style="flex:1;padding:10px;border:none;outline:none"/>
      <button id="send-btn" type="submit"
        style="padding:10px 14px;border:none;background:#000;color:#fff;cursor:pointer">
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
  const input = box.querySelector("#chat-input");
  const sendBtn = box.querySelector("#send-btn");

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
  }

  function showTyping() {
    const el = document.createElement("div");
    el.innerText = "...";
    msgs.appendChild(el);
    msgs.scrollTop = msgs.scrollHeight;
    return el;
  }

  function blockInput() {
    input.disabled = true;
    input.placeholder = "Please fill the form to continue...";
    sendBtn.disabled = true;
    sendBtn.style.opacity = "0.5";
  }

  function unblockInput() {
    input.disabled = false;
    input.placeholder = "Type your question...";
    sendBtn.disabled = false;
    sendBtn.style.opacity = "1";
    input.focus();
  }

  // ---------------- LEAD FORM ----------------
  function showLeadForm() {
    awaitingLead = true;
    blockInput();

    const title = leadConfig?.form_title || "Before we continue...";
    const subtitle = leadConfig?.form_subtitle || "Please share your details to keep chatting.";

    // Overlay sits inside msgs container
    const overlay = document.createElement("div");
    overlay.id = "lead-overlay";
    overlay.style.cssText = `
      position:absolute;inset:0;
      background:rgba(255,255,255,0.97);
      display:flex;flex-direction:column;
      align-items:center;justify-content:center;
      padding:24px;z-index:10;
    `;

    overlay.innerHTML = `
      <div style="width:100%;max-width:280px;text-align:center;">
        <div style="font-size:22px;margin-bottom:8px;">👋</div>
        <h3 style="font-size:15px;font-weight:600;margin:0 0 6px;">${title}</h3>
        <p style="font-size:12px;color:#666;margin:0 0 18px;">${subtitle}</p>

        <div style="display:flex;flex-direction:column;gap:8px;text-align:left;">
          <input id="lf-name" type="text" placeholder="Your name *" style="
            border:1px solid #ddd;border-radius:8px;
            padding:9px 12px;font-size:13px;
            width:100%;box-sizing:border-box;outline:none;
          "/>
          <input id="lf-email" type="email" placeholder="Email address *" style="
            border:1px solid #ddd;border-radius:8px;
            padding:9px 12px;font-size:13px;
            width:100%;box-sizing:border-box;outline:none;
          "/>
          <input id="lf-phone" type="tel" placeholder="Phone number *" style="
            border:1px solid #ddd;border-radius:8px;
            padding:9px 12px;font-size:13px;
            width:100%;box-sizing:border-box;outline:none;
          "/>
          <div id="lf-error" style="
            color:#e53e3e;font-size:11px;display:none;
          "></div>
          <button id="lf-submit" style="
            background:#000;color:#fff;border:none;
            border-radius:8px;padding:10px;
            font-size:13px;font-weight:500;
            cursor:pointer;margin-top:2px;
          ">
            Continue chatting →
          </button>
        </div>
      </div>
    `;

    msgs.style.position = "relative";
    msgs.appendChild(overlay);

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

          addMsg("assistant", `Thanks ${name}! How can I help you?`);

          // Answer the question they had asked before form appeared
          if (pendingQuestion) {
            await askBot(pendingQuestion);
            pendingQuestion = null;
          }
        } else {
          submitBtn.textContent = "Continue chatting →";
          submitBtn.disabled = false;
          errorEl.textContent = "Something went wrong. Please try again.";
          errorEl.style.display = "block";
        }
      } catch (e) {
        submitBtn.textContent = "Continue chatting →";
        submitBtn.disabled = false;
        errorEl.textContent = "Network error. Please try again.";
        errorEl.style.display = "block";
      }
    };
  }

  // ---------------- ASK BOT ----------------
  async function askBot(question) {
    const typing = showTyping();

    const res = await fetch(`${apiBase}/public/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        projectId,
        message: question,
        userId,
      }),
    });

    const data = await res.json();
    typing.remove();
    addMsg("assistant", render(data.answer || ""));
  }

  // ---------------- FORM SUBMIT ----------------
  form.onsubmit = async (e) => {
    e.preventDefault();

    const text = input.value.trim();
    if (!text || awaitingLead) return;

    input.value = "";
    addMsg("user", render(text));
    userMessageCount++;

    const threshold = leadConfig?.trigger_after_messages ?? 2;

    // Show lead form if enabled, not yet captured, and threshold reached
    if (leadConfig && !lead && userMessageCount >= threshold) {
      pendingQuestion = text;
      showLeadForm();
      return;
    }

    await askBot(text);
  };

  // ---------------- INIT ----------------
  fetchLeadConfig();
})();