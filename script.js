const form = document.querySelector("#chatForm");
const input = document.querySelector("#promptInput");
const messages = document.querySelector("#messages");
const statusEl = document.querySelector("#status");
const API_BASE = window.location.protocol === "file:" ? "http://127.0.0.1:8000" : "";

function addMessage(role, text, imageUrl) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  if (imageUrl) {
    const image = document.createElement("img");
    image.src = imageUrl;
    image.alt = text;
    bubble.appendChild(image);
  }

  article.appendChild(bubble);
  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
}

async function sendPrompt(prompt) {
  addMessage("user", prompt);
  statusEl.textContent = "Thinking";
  form.querySelector("button").disabled = true;

  try {
    const response = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
    const data = await response.json();

    if (!response.ok || data.error) {
      throw new Error(data.error || "Request failed.");
    }

    if (data.type === "image") {
      const imageUrl = data.image_url.startsWith("http") ? data.image_url : `${API_BASE}${data.image_url}`;
      addMessage("assistant", `Generated ${data.landmark.replaceAll("_", " ")}.`, imageUrl);
    } else {
      addMessage("assistant", data.answer || "I could not generate an answer.");
    }
  } catch (error) {
    addMessage("assistant", error.message);
  } finally {
    statusEl.textContent = "Ready";
    form.querySelector("button").disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const prompt = input.value.trim();
  if (!prompt) return;
  input.value = "";
  sendPrompt(prompt);
});

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.textContent;
    sendPrompt(button.textContent);
  });
});
