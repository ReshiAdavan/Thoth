import "./style.css";
import { TokenizerClient } from "./worker-client.js";
import { mapText } from "./unicode.js";
const $ = (id) => document.getElementById(id),
  client = new TokenizerClient();
const colors = [
  "#b9a1ed",
  "#90bedb",
  "#add4bb",
  "#e5c48c",
  "#dbacc2",
  "#b6c4ee",
];
let ids = [],
  mapping = null,
  timer,
  revision = 0;
const samples = {
  code: 'def hello(name):\n    return f"Hello, {name}!"',
  unicode: "Hello · Bonjour · 你好 · مرحبا · नमस्ते 🦉",
  spaces: "one two  three\n\tfour\n\n",
};
const localModels = {};
let localAbort;
function stop() {
  localAbort?.abort();
  clearTimeout(timer);
  client.cancel();
  revision++;
}
function invalidate() {
  ids = [];
  mapping = null;
  $("copy").disabled = true;
  $("count").textContent = "—";
  $("tokens").replaceChildren();
  $("readable").replaceChildren();
  $("results").replaceChildren();
  $("detail").textContent =
    "Select text or a token to inspect its bytes and position.";
}
function failure(e) {
  if (e.name !== "AbortError") $("status").textContent = e.message;
}
function highlight(indices) {
  document
    .querySelectorAll("#tokens button")
    .forEach((b, i) => b.classList.toggle("selected", indices.includes(i)));
  document.querySelectorAll("#readable span").forEach((span, i) =>
    span.classList.toggle(
      "selected",
      mapping.segments[i].indices.some((n) => indices.includes(n)),
    ),
  );
}
function inspect(index, bytes) {
  const range = mapping.tokens[index],
    segments = mapping.segments.filter((s) => s.indices.includes(index));
  highlight([index]);
  if (segments.length) {
    $("text").focus();
    $("text").setSelectionRange(segments[0].start, segments.at(-1).end);
  }
  $("detail").textContent =
    `Token ${index + 1} / ${ids.length} · ID ${ids[index]} · UTF-8 bytes ${range.start}–${range.end} (end exclusive) · [${bytes.join(", ")}]`;
}
function render(result, text) {
  ids = result.ids;
  mapping = mapText(text, result.bytes);
  $("tokens").replaceChildren();
  $("readable").replaceChildren();
  mapping.segments.forEach((segment) => {
    const span = document.createElement("span");
    span.textContent = segment.text;
    span.dataset.tokens = segment.indices.join(",");
    span.title = `Tokens ${segment.indices.map((i) => i + 1).join(", ")} · bytes ${segment.byteStart}–${segment.byteEnd}`;
    span.onclick = () => {
      highlight(segment.indices);
      $("text").focus();
      $("text").setSelectionRange(segment.start, segment.end);
      $("detail").textContent =
        `${JSON.stringify(segment.text)} · token IDs [${segment.indices.map((i) => ids[i]).join(", ")}] · UTF-8 bytes ${segment.byteStart}–${segment.byteEnd}`;
    };
    $("readable").append(span);
  });
  result.bytes.forEach((bytes, i) => {
    let label;
    try {
      label = new TextDecoder("utf-8", { fatal: true }).decode(
        new Uint8Array(bytes),
      );
    } catch {
      label = bytes.map((b) => b.toString(16).padStart(2, "0")).join(" ");
    }
    const b = document.createElement("button");
    b.className = "token";
    b.style.background = colors[i % colors.length];
    b.textContent = $("whitespace").checked
      ? label.replaceAll(" ", "·").replaceAll("\n", "↵\n").replaceAll("\t", "⇥")
      : label;
    b.setAttribute("aria-label", `Token ${i + 1}, ID ${ids[i]}`);
    b.title = `ID ${ids[i]} · bytes ${mapping.tokens[i].start}–${mapping.tokens[i].end}`;
    b.onclick = () => inspect(i, bytes);
    $("tokens").append(b);
  });
  $("count").textContent = ids.length;
  $("copy").disabled = false;
  $("status").textContent =
    `${text.length} UTF-16 units · ${new TextEncoder().encode(text).length} UTF-8 bytes · ${ids.length} tokens`;
  renderCounts(result.counts);
}
function renderCounts(counts) {
  $("results").replaceChildren();
  for (const [name, count] of Object.entries(counts)) {
    const div = document.createElement("div");
    div.className = "result";
    div.textContent = name;
    const strong = document.createElement("strong");
    strong.textContent = count;
    div.append(strong);
    $("results").append(div);
  }
}
async function update() {
  stop();
  const current = revision,
    text = $("text").value;
  invalidate();
  const name = $("encoding").value,
    local = name.startsWith("local-");
  $("local-controls").hidden = !local;
  $("local-train").hidden = ["local-gpt", "local-trained"].includes(name);
  $("sp-upload-label").hidden = name !== "local-llama";
  $("compare").disabled = false;
  $("local-state").textContent = localModels[name]
    ? "Using your trained/imported session model."
    : "Saved 16,384-token English + code vocabulary. Ready to use; no training required.";
  $("comparison").hidden = !$("compare").checked;
  $("status").textContent = "Tokenizing…";
  try {
    if (local) {
      const result = await localRequest({
        operation: "encode",
        name,
        text,
        modelId: localModels[name],
        compare: $("compare").checked,
      });
      if (current === revision)
        result.pieces ? renderPieces(result) : render(result, text);
      return;
    }
    const result = await client.request({
      operation: "encode",
      text,
      name: $("encoding").value,
      compare: $("compare").checked,
    });
    if (current === revision) render(result, text);
  } catch (e) {
    if (current === revision) failure(e);
  }
}
$("text").oninput = () => {
  stop();
  invalidate();
  $("status").textContent = "Waiting for input…";
  timer = setTimeout(update, 120);
};
for (const id of ["encoding", "whitespace", "compare"]) $(id).onchange = update;
$("clear").onclick = () => {
  $("text").value = "";
  update();
};
document.querySelectorAll("[data-sample]").forEach(
  (b) =>
    (b.onclick = () => {
      $("text").value = samples[b.dataset.sample];
      update();
    }),
);
$("text").onselect = () => {
  if (!mapping) return;
  const start = $("text").selectionStart,
    end = $("text").selectionEnd;
  if (start === end) return;
  highlight([
    ...new Set(
      mapping.segments
        .filter((s) => s.start < end && s.end > start)
        .flatMap((s) => s.indices),
    ),
  ]);
};
$("copy").onclick = async () => {
  try {
    await navigator.clipboard.writeText(JSON.stringify(ids));
    $("status").textContent = "Token IDs copied";
  } catch {
    $("status").textContent = "Clipboard unavailable in this browser";
  }
};
$("cancel").onclick = () => {
  stop();
  $("status").textContent =
    "Operation canceled. Edit text or change encoding to retry.";
};
update();

async function localRequest(data) {
  localAbort = new AbortController();
  const response = await fetch("/api/tokenize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
    signal: localAbort.signal,
  });
  let result;
  try {
    result = await response.json();
  } catch {
    throw Error(
      "Start the local backend: python -m thoth_demo (from the project virtual environment).",
    );
  }
  if (!response.ok)
    throw Error(result.error || "Local tokenizer request failed");
  return result;
}
function renderPieces(result) {
  renderCounts(result.counts);
  ids = result.ids;
  mapping = null;
  $("tokens").replaceChildren();
  $("readable").textContent = result.decoded;
  result.pieces.forEach((piece, i) => {
    const button = document.createElement("button");
    button.className = "token";
    button.style.background = colors[i % colors.length];
    button.textContent = piece;
    button.onclick = () => {
      $("detail").textContent =
        `SentencePiece ID ${ids[i]} · piece ${JSON.stringify(piece)}. Model normalization means these are not byte offsets into the original input.`;
    };
    $("tokens").append(button);
  });
  $("count").textContent = ids.length;
  $("copy").disabled = false;
  $("status").textContent =
    `${ids.length} tokens · SentencePiece decoded output shown above; normalization and unknown-token behavior follow the model.`;
}
$("local-train").onclick = async () => {
  stop();
  const current = revision,
    name = $("encoding").value;
  invalidate();
  $("status").textContent = "Training local implementation…";
  try {
    const result = await localRequest({
      operation: "train",
      name,
      text: $("text").value,
      size: Number($("local-size").value),
    });
    if (current !== revision) return;
    localModels[name] = result.modelId;
    await update();
  } catch (e) {
    if (current === revision) failure(e);
  }
};
$("sp-upload").onchange = async (e) => {
  stop();
  const current = revision;
  try {
    const file = e.target.files[0];
    if (!file) return;
    if (file.size > 1_000_000) throw Error("Model exceeds 1 MB");
    const bytes = new Uint8Array(await file.arrayBuffer());
    let binary = "";
    for (const byte of bytes) binary += String.fromCharCode(byte);
    if (current !== revision) return;
    const result = await localRequest({
      operation: "import",
      name: "local-llama",
      model: btoa(binary),
    });
    if (current !== revision) return;
    localModels["local-llama"] = result.modelId;
    await update();
  } catch (error) {
    if (current === revision) failure(error);
  } finally {
    e.target.value = "";
  }
};
