import { Tiktoken } from "js-tiktoken/lite";
let encoding;
self.onmessage = async ({ data }) => {
  const { id, text = "", name } = data;
  try {
    if (name !== "cl100k_base")
      throw Error("Only cl100k_base is available as a browser reference");
    if (new TextEncoder().encode(text).length > 20000)
      throw Error(
        "Please use at most 20,000 UTF-8 bytes for this interactive preview.",
      );
    if (!encoding)
      encoding = import("js-tiktoken/ranks/cl100k_base").then(
        (m) => new Tiktoken(m.default),
      );
    const t = await encoding,
      ids = t.encode(text, [], []);
    self.postMessage({
      id,
      result: {
        ids,
        bytes: ids.map((i) => Array.from(t.textMap.get(i))),
        counts: data.compare ? { cl100k_base: ids.length } : {},
      },
    });
  } catch (error) {
    encoding = null;
    self.postMessage({ id, error: error.message });
  }
};
