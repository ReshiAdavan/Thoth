import { test, expect } from "@playwright/test";
test.skip(!process.env.THOTH_LOCAL_TESTS, "Requires local backend");
for (const name of ["local-basic", "local-rust", "local-regex", "local-llama"])
  test(`${name} loads saved vocabulary and compares`, async ({ page }) => {
    await page.goto("/");
    await page.locator("#encoding").selectOption(name);
    await page.locator("#text").fill("Hello world! Python programming is fun.");
    await expect(page.locator("#readable")).toHaveText(
      "Hello world! Python programming is fun.",
    );
    await expect(page.locator("#local-state")).toContainText("Ready to use");
    expect(Number(await page.locator("#count").textContent())).toBeLessThan(25);
    await page.locator("#compare").check();
    await expect(page.locator(".result")).toHaveCount(2);
    await expect(page.locator("#results")).toContainText("cl100k_base");
  });
test("temporary training remains separate from saved model", async ({
  page,
}) => {
  await page.goto("/");
  await page.locator("#encoding").selectOption("local-basic");
  await page.locator("#text").fill("abababab");
  await page.locator("#local-size").fill("258");
  await page.locator("#local-train").click();
  await expect(page.locator("#count")).toHaveText("2");
  await expect(page.locator("#local-state")).toContainText("session model");
  await page.reload();
  await page.locator("#encoding").selectOption("local-basic");
  await expect(page.locator("#local-state")).toContainText("Ready to use");
  await page.locator("#text").fill("Hello world! Python programming is fun.");
  await expect(page.locator("#readable")).toHaveText(
    "Hello world! Python programming is fun.",
  );
  expect(Number(await page.locator("#count").textContent())).toBeLessThan(25);
});
