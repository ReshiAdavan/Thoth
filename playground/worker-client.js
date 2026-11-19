// Interrupt synchronous BPE work by terminating its worker; idle workers retain tables.
export class TokenizerClient {
  constructor() {
    this.sequence = 0;
    this.worker = null;
    this.pending = null;
  }
  cancel() {
    if (this.pending) {
      this.pending.reject(new DOMException("Superseded", "AbortError"));
      this.pending = null;
      this.worker.terminate();
      this.worker = null;
    }
  }
  request(payload) {
    this.cancel();
    if (!this.worker) {
      this.worker = new Worker(
        new URL("./tokenizer.worker.js", import.meta.url),
        { type: "module" },
      );
      this.worker.onmessage = ({ data }) => {
        if (data.id !== this.pending?.id) return;
        const p = this.pending;
        this.pending = null;
        data.error ? p.reject(Error(data.error)) : p.resolve(data.result);
      };
      this.worker.onerror = () => {
        this.pending?.reject(Error("Tokenizer worker failed. Please retry."));
        this.pending = null;
        this.worker.terminate();
        this.worker = null;
      };
    }
    return new Promise((resolve, reject) => {
      const id = ++this.sequence;
      this.pending = { id, resolve, reject };
      this.worker.postMessage({ ...payload, id });
    });
  }
}
