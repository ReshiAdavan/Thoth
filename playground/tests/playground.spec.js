import { test, expect } from "@playwright/test";
import { getEncoding } from "js-tiktoken";
test("one JavaScript reference with readable Unicode and inspection", async ({
  page,
}) => {
  await page.goto("/");
  await page.locator("#encoding").selectOption("cl100k_base");
  expect(await page.locator("#encoding option").count()).toBe(5);
  await page.locator("#text").fill("Hello 🦉 中文\n");
  await expect(page.locator("#count")).toHaveText(
    String(getEncoding("cl100k_base").encode("Hello 🦉 中文\n").length),
  );
  await expect(page.locator("#readable")).toHaveText("Hello 🦉 中文\n");
  await expect(page.locator("#tokens")).not.toContainText("�");
  await page.locator("#tokens button").first().click();
  await expect(page.locator("#detail")).toContainText("UTF-8 bytes");
  await page.locator("#whitespace").check();
  await expect(page.locator("#tokens")).toContainText("↵");
  await page.locator("#clear").click();
  await expect(page.locator("#count")).toHaveText("0");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
});
test("ordinary special text and no extra encoding downloads", async ({
  page,
}) => {
  const requests = [];
  page.on("request", (r) => requests.push(r.url()));
  await page.goto("/");
  await page.locator("#encoding").selectOption("cl100k_base");
  await page.locator("#text").fill("<|endoftext|>");
  await expect(page.locator("#count")).toHaveText(
    String(getEncoding("cl100k_base").encode("<|endoftext|>", [], []).length),
  );
  expect(requests.some((u) => /o200k_base|p50k_base|r50k_base/.test(u))).toBe(
    false,
  );
});
