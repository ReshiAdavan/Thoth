import { test, expect } from "@playwright/test";
test.skip(
  !process.env.THOTH_LOCAL_TESTS,
  "Requires local Python/native backend",
);
for (const name of ["local-basic", "local-regex", "local-rust"])
  test(`${name} runs real training`, async ({ page }) => {
    await page.goto("/");
    await page.locator("#encoding").selectOption(name);
    await page.locator("#text").fill("abababab");
    await page.locator("#local-size").fill("258");
    await page.locator("#local-train").click();
    await expect(page.locator("#count")).toHaveText("2");
    await expect(page.locator("#local-state")).toContainText("session model");
    await expect(page.locator("#readable")).toHaveText("abababab");
  });
test("Python GPT adapter and SentencePiece training", async ({ page }) => {
  await page.goto("/");
  await page.locator("#encoding").selectOption("local-gpt");
  await page.locator("#text").fill("Hello");
  await expect(page.locator("#count")).toHaveText("1");
  await page.locator("#encoding").selectOption("local-llama");
  await expect(page.locator("#status")).toContainText("model file");
  await page.locator("#text").fill("hello hello world");
  await page.locator("#local-size").fill("256");
  await page.locator("#local-train").click();
  await expect(page.locator("#status")).toContainText("SentencePiece decoded");
  await expect(page.locator("#readable")).toHaveText("hello hello world");
});

test("saved Thoth model is ready without training", async ({ page }) => {
  await page.goto("/");
  await page.locator("#encoding").selectOption("local-trained");
  await page.locator("#text").fill("Hello world! Python programming is fun.");
  await expect(page.locator("#readable")).toHaveText(
    "Hello world! Python programming is fun.",
  );
  await expect(page.locator("#local-train")).toBeHidden();
  await expect(page.locator("#local-state")).toContainText("Ready to use");
  expect(Number(await page.locator("#count").textContent())).toBeLessThan(20);
});
