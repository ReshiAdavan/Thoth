import "./style.css";
import { getEncoding } from "js-tiktoken";
import { BPE } from "./bpe";
const $ = (id) => document.getElementById(id),
  colors = ["#b9a1ed", "#90bedb", "#add4bb", "#e5c48c", "#dbacc2", "#b6c4ee"];
let cache = {},
  custom = new BPE(),
  ids = [],
  timer,
  playTimer;
function trace() {
  let n = Number($("step").value),
    t = new BPE({ ...custom.export(), merges: custom.merges.slice(0, n) });
  $("trace").textContent =
    `Step ${n}/${custom.merges.length}: [${t.encode($("text").value).join(", ")}]`;
}
const samples = {
  code: 'def hello(name):\n    return f"Hello, {name}!"',
  unicode: "Hello · Bonjour · 你好 · مرحبا · नमस्ते 🦉",
  spaces: "one two  three\n\tfour\n\n",
};
function tokenizer(name) {
  return name === "custom" ? custom : (cache[name] ??= getEncoding(name));
}
function update() {
  clearInterval(playTimer);
  $("detail").textContent =
    "Select a token to inspect its ID, bytes and position.";
  try {
    let text = $("text").value;
    if (new TextEncoder().encode(text).length > 20000)
      throw Error(
        "Please use at most 20,000 UTF-8 bytes for this interactive preview.",
      );
    let name = $("encoding").value,
      t = tokenizer(name);
    ids = t.encode(text);
    $("count").textContent = ids.length;
    $("tokens").replaceChildren();
    $("training").hidden = name !== "custom";
    const decoder = new TextDecoder();
    ids.forEach((id, i) => {
      let bytes =
        name === "custom"
          ? t.vocab[id]
          : Array.from(
              t.textMap.get(id) ?? new TextEncoder().encode(t.decode([id])),
            );
      let label = decoder.decode(new Uint8Array(bytes));
      let b = document.createElement("button");
      b.className = "token";
      b.style.background = colors[i % colors.length];
      b.textContent = $("whitespace").checked
        ? label
            .replaceAll(" ", "·")
            .replaceAll("\n", "↵\n")
            .replaceAll("\t", "⇥")
        : label;
      b.setAttribute("aria-label", `Token ${i + 1}, ID ${id}`);
      b.title = `ID ${id} · ${bytes.length} bytes`;
      b.onclick = () => {
        $("detail").textContent =
          `Token ${i + 1} / ${ids.length} · ID ${id} · bytes [${bytes.join(", ")}] · text ${JSON.stringify(label)}`;
      };
      $("tokens").append(b);
    });
    $("status").textContent =
      `${text.length} UTF-16 units · ${new TextEncoder().encode(text).length} UTF-8 bytes · ${ids.length} tokens`;
    $("comparison").hidden = !$("compare").checked;
    if ($("compare").checked) {
      $("results").replaceChildren();
      for (let n of ["cl100k_base", "o200k_base", "p50k_base", "r50k_base"]) {
        let div = document.createElement("div");
        div.className = "result";
        div.textContent = n;
        let count = document.createElement("strong");
        count.textContent = tokenizer(n).encode(text).length;
        div.append(count);
        $("results").append(div);
      }
    }
  } catch (e) {
    ids = [];
    $("results").replaceChildren();
    $("status").textContent = e.message;
    $("tokens").replaceChildren();
    $("count").textContent = "—";
  }
}
$("text").oninput = () => {
  clearTimeout(timer);
  timer = setTimeout(update, 120);
};
for (let id of ["encoding", "whitespace", "compare"]) $(id).onchange = update;
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
$("copy").onclick = async () => {
  try {
    await navigator.clipboard.writeText(JSON.stringify(ids));
    $("status").textContent = "Token IDs copied";
  } catch {
    $("status").textContent = "Clipboard unavailable in this browser";
  }
};
$("train").onclick = () => {
  try {
    if (new TextEncoder().encode($("text").value).length > 20000)
      throw Error("Training input exceeds 20,000 bytes");
    custom.train($("text").value, Number($("vocab").value));
    $("merges").textContent =
      `Learned ${custom.merges.length} merges · ${custom.vocab.length} vocabulary entries`;
    $("step").max = custom.merges.length;
    $("step").value = 0;
    trace();
    update();
  } catch (e) {
    $("status").textContent = e.message;
  }
};
$("export").onclick = () => {
  let url = URL.createObjectURL(
    new Blob([JSON.stringify(custom.export(), null, 2)], {
      type: "application/json",
    }),
  );
  let a = document.createElement("a");
  a.href = url;
  a.download = "thoth-model.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$("import").onchange = async (e) => {
  try {
    let file = e.target.files[0];
    if (!file) return;
    if (file.size > 1000000) throw Error("Model file exceeds 1 MB");
    custom = new BPE(JSON.parse(await file.text()));
    $("merges").textContent = `Imported ${custom.merges.length} merges`;
    $("step").max = custom.merges.length;
    $("step").value = 0;
    trace();
    update();
  } catch (error) {
    $("status").textContent = error.message;
  }
};
update();

$("step").oninput = trace;
$("play").onclick = () => {
  clearInterval(playTimer);
  $("step").value = 0;
  trace();
  playTimer = setInterval(() => {
    if (Number($("step").value) >= custom.merges.length) {
      clearInterval(playTimer);
      return;
    }
    $("step").value = Number($("step").value) + 1;
    trace();
  }, 500);
};
