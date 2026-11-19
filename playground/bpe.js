export class BPE {
  constructor(data = { version: 1, kind: "basic", merges: [] }) {
    if (
      data.version !== 1 ||
      data.kind !== "basic" ||
      !Array.isArray(data.merges)
    )
      throw Error("Expected a version 1 basic Thoth model");
    this.vocab = Array.from({ length: 256 }, (_, i) => [i]);
    this.merges = [];
    let seen = new Set();
    for (const p of data.merges) {
      if (
        !Array.isArray(p) ||
        p.length !== 2 ||
        p.some(
          (i) => !Number.isInteger(i) || i < 0 || i >= this.vocab.length,
        ) ||
        seen.has(p.join(","))
      )
        throw Error("Invalid merge");
      seen.add(p.join(","));
      this.merges.push(p);
      this.vocab.push([...this.vocab[p[0]], ...this.vocab[p[1]]]);
    }
  }
  merge(ids, p, id) {
    let out = [];
    for (let i = 0; i < ids.length; i++) {
      if (ids[i] === p[0] && ids[i + 1] === p[1]) {
        out.push(id);
        i++;
      } else out.push(ids[i]);
    }
    return out;
  }
  train(text, size) {
    if (!text || !Number.isInteger(size) || size < 256 || size > 1024)
      throw Error("Use nonempty text and a vocabulary from 256 to 1024");
    this.vocab = Array.from({ length: 256 }, (_, i) => [i]);
    this.merges = [];
    let ids = Array.from(new TextEncoder().encode(text));
    while (this.vocab.length < size && ids.length > 1) {
      let counts = new Map();
      for (let i = 0; i < ids.length - 1; i++) {
        let k = ids.slice(i, i + 2).join(",");
        counts.set(k, (counts.get(k) || 0) + 1);
      }
      let pairs = [...counts]
        .map(([k, n]) => ({ p: k.split(",").map(Number), n }))
        .sort((a, b) => b.n - a.n || a.p[0] - b.p[0] || a.p[1] - b.p[1]);
      let p = pairs[0].p;
      ids = this.merge(ids, p, this.vocab.length);
      this.merges.push(p);
      this.vocab.push([...this.vocab[p[0]], ...this.vocab[p[1]]]);
    }
  }
  encode(text) {
    let ids = Array.from(new TextEncoder().encode(text));
    this.merges.forEach((p, i) => (ids = this.merge(ids, p, i + 256)));
    return ids;
  }
  export() {
    return { version: 1, kind: "basic", merges: this.merges };
  }
}
