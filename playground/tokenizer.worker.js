import { Tiktoken } from "js-tiktoken/lite";
import { BPE } from "./bpe.js";
const loaders = {
  cl100k_base: () => import("js-tiktoken/ranks/cl100k_base"),
  o200k_base: () => import("js-tiktoken/ranks/o200k_base"),
  p50k_base: () => import("js-tiktoken/ranks/p50k_base"),
  r50k_base: () => import("js-tiktoken/ranks/r50k_base"),
};
const cache = new Map();
async function tokenizer(name, model) {
  if (name === "custom") return new BPE(model);
  if (!loaders[name]) throw Error("Unknown encoding");
  if (!cache.has(name))
    cache.set(
      name,
      loaders[name]().then((m) => new Tiktoken(m.default)),
    );
  try {
    return await cache.get(name);
  } catch (e) {
    cache.delete(name);
    throw e;
  }
}
self.onmessage = async ({ data }) => {
  const { id, operation, text = "", name, model, compare, size, step } = data;
  try {
    if (new TextEncoder().encode(text).length > 20000)
      throw Error(
        "Please use at most 20,000 UTF-8 bytes for this interactive preview.",
      );
    let result;
    if (operation === "import")
      result = new BPE(JSON.parse(data.json)).export();
    else if (operation === "train") {
      const t = new BPE();
      t.train(text, size);
      result = t.export();
    } else if (operation === "trace")
      result = new BPE({
        ...model,
        merges: model.merges.slice(0, step),
      }).encode(text);
    else {
      const t = await tokenizer(name, model),
        ids = t.encode(text, [], []);
      const bytes = ids.map((id) =>
        Array.from(name === "custom" ? t.vocab[id] : t.textMap.get(id)),
      );
      const counts = {};
      if (compare)
        for (const n of Object.keys(loaders))
          counts[n] = (await tokenizer(n)).encode(text, [], []).length;
      result = { ids, bytes, counts };
    }
    self.postMessage({ id, result });
  } catch (e) {
    self.postMessage({ id, error: e.message });
  }
};
