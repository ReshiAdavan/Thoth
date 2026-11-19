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
let model = { version: 1, kind: "basic", merges: [] },
  ids = [],
  mapping = null,
  timer,
  playTimer,
  revision = 0;
const samples = {
  code: 'def hello(name):\n    return f"Hello, {name}!"',
  unicode: "Hello · Bonjour · 你好 · مرحبا · नमस्ते 🦉",
  spaces: "one two  three\n\tfour\n\n",
};
function stop() {
  clearTimeout(timer);
  clearInterval(playTimer);
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
  $("results").replaceChildren();
  for (const [name, count] of Object.entries(result.counts)) {
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
  $("training").hidden = $("encoding").value !== "custom";
  $("comparison").hidden = !$("compare").checked;
  $("status").textContent = "Tokenizing…";
  try {
    const result = await client.request({
      operation: "encode",
      text,
      name: $("encoding").value,
      model,
      compare: $("compare").checked,
    });
    if (current === revision) render(result, text);
  } catch (e) {
    failure(e);
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
async function trace() {
  const current = revision,
    step = Number($("step").value);
  try {
    const tokens = await client.request({
      operation: "trace",
      text: $("text").value,
      model,
      step,
    });
    if (current === revision)
      $("trace").textContent =
        `Step ${step}/${model.merges.length}: [${tokens.join(", ")}]`;
  } catch (e) {
    failure(e);
  }
}
function modelChanged(message) {
  $("merges").textContent = message;
  $("step").max = model.merges.length;
  $("step").value = 0;
  $("trace").textContent = "";
  return update();
}
$("train").onclick = async () => {
  stop();
  const current = revision;
  $("status").textContent = "Training…";
  try {
    const next = await client.request({
      operation: "train",
      text: $("text").value,
      size: Number($("vocab").value),
    });
    if (current !== revision) return;
    model = next;
    await modelChanged(
      `Learned ${model.merges.length} merges · ${256 + model.merges.length} vocabulary entries`,
    );
  } catch (e) {
    failure(e);
  }
};
$("import").onchange = async (e) => {
  stop();
  const current = revision;
  try {
    const file = e.target.files[0];
    if (!file) return;
    if (file.size > 1_000_000) throw Error("Model file exceeds 1 MB");
    const json = await file.text();
    if (current !== revision) return;
    const next = await client.request({ operation: "import", json });
    if (current !== revision) return;
    model = next;
    await modelChanged(`Imported ${model.merges.length} merges`);
  } catch (e) {
    failure(e);
  } finally {
    e.target.value = "";
  }
};
$("export").onclick = () => {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(model, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = "thoth-model.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$("step").oninput = () => {
  clearInterval(playTimer);
  trace();
};
$("play").onclick = async () => {
  clearInterval(playTimer);
  $("step").value = 0;
  await trace();
  playTimer = setInterval(async () => {
    if (Number($("step").value) >= model.merges.length) {
      clearInterval(playTimer);
      return;
    }
    $("step").value = Number($("step").value) + 1;
    await trace();
  }, 500);
};
$("cancel").onclick = () => {
  stop();
  $("status").textContent =
    "Operation canceled. Edit text or change encoding to retry.";
};
update();
