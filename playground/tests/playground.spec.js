import { test, expect } from "@playwright/test";
import { getEncoding } from "js-tiktoken";
import { BPE } from "../bpe.js";
import { readFile } from "node:fs/promises";
test("text, encoding selection, whitespace, inspection, comparison and clear", async ({
  page,
}) => {
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await page.locator("#text").fill("Hello 🦉 中文\n");
  await expect(page.locator("#count")).toHaveText(
    String(getEncoding("cl100k_base").encode("Hello 🦉 中文\n").length),
  );
  await page.locator("#tokens button").first().click();
  await expect(page.locator("#detail")).toContainText("ID");
  await page.locator("#whitespace").check();
  await expect(page.locator("#tokens")).toContainText("↵");
  await page.locator("#encoding").selectOption("o200k_base");
  await expect(page.locator("#count")).toHaveText(
    String(getEncoding("o200k_base").encode("Hello 🦉 中文\n").length),
  );
  await page.locator("#compare").check();
  await expect(page.locator(".result")).toHaveCount(4);
  await page.locator("#clear").click();
  await expect(page.locator("#count")).toHaveText("0");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  expect(errors).toEqual([]);
});
test("train, export, import and reject malformed models", async ({ page }) => {
  await page.goto("/");
  await page.locator("#encoding").selectOption("custom");
  await page.locator("#text").fill("abababab");
  await page.locator("#vocab").fill("258");
  await page.locator("#train").click();
  await expect(page.locator("#count")).toHaveText("2");
  const download = page.waitForEvent("download");
  await page.locator("#export").click();
  const file = await download;
  expect(file.suggestedFilename()).toBe("thoth-model.json");
  expect(JSON.parse(await readFile(await file.path(), "utf8"))).toEqual({
    version: 1,
    kind: "basic",
    merges: [
      [97, 98],
      [256, 256],
    ],
  });
  await page.locator("#play").click();
  await expect(page.locator("#trace")).toHaveText("Step 2/2: [257, 257]");
  await page.locator("#import").setInputFiles({
    name: "model.json",
    mimeType: "application/json",
    buffer: Buffer.from(
      JSON.stringify({
        version: 1,
        kind: "basic",
        merges: [
          [97, 98],
          [256, 256],
        ],
      }),
    ),
  });
  await expect(page.locator("#merges")).toContainText("Imported 2");
  await page.locator("#import").setInputFiles({
    name: "bad.json",
    mimeType: "application/json",
    buffer: Buffer.from("{}"),
  });
  await expect(page.locator("#status")).toContainText("Expected a version 1");
});
test("BPE deterministic and lossless", () => {
  let t = new BPE();
  t.train("abababab", 258);
  expect(t.encode("abababab")).toEqual([257, 257]);
  expect(new BPE(t.export()).encode("abababab")).toEqual([257, 257]);
  let text = "你好 🦉";
  expect(
    new TextDecoder().decode(
      new Uint8Array(t.encode(text).flatMap((i) => t.vocab[i])),
    ),
  ).toBe(text);
});

test("special-token-looking text remains ordinary text", async ({ page }) => {
  await page.goto("/");
  await page.locator("#text").fill("<|endoftext|>");
  await expect(page.locator("#count")).toHaveText(
    String(getEncoding("cl100k_base").encode("<|endoftext|>", [], []).length),
  );
  await page.locator("#compare").check();
  await expect(page.locator(".result")).toHaveCount(4);
});
