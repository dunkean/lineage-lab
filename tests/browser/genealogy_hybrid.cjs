const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs");

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.STUDIO_BROWSER ||
      "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    headless: true,
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("response", (response) => {
      if (response.status() >= 400) errors.push(response.status() + " " + response.url());
    });
    await page.goto(process.env.STUDIO_URL || "http://127.0.0.1:8784");
    await page.waitForFunction(() => document.querySelector("#treeNote").textContent);
    await page.click("#tabConfig");
    await page.selectOption("#preset", "hybrid");
    assert.equal(await page.locator("#populationModel").inputValue(), "hybrid");
    fs.mkdirSync("output/genealogy/screenshots", { recursive: true });
    await page.screenshot({ path: "output/genealogy/screenshots/hybrid-config.png", fullPage: true });
    // Exercise generation, automatic opening, selection and opaque-ID searching.
    await page.fill("#founders", "300");
    await page.click("#generate");
    await page.waitForFunction(() => !document.querySelector("#openResult").hidden,
      { timeout: 60000 });
    await page.waitForFunction(() =>
      document.querySelector("#configView").hidden &&
      document.querySelector("#diagnostics").textContent.includes("Hybrid"));
    await page.click("#tabIndividual");
    const identity = await page.locator("#identity").inputValue();
    assert.match(identity, /^h:[a-f0-9]{64}:\d+:\d+$/);
    await page.fill("#identity", identity);
    await page.locator("#search button.primary").click();
    await page.selectOption("#direction", "descendants");
    await page.waitForFunction(() => !document.querySelector("#treeNote").textContent.includes("Loading"));
    assert.equal(await page.locator("#error").isVisible(), false);
    await page.screenshot({ path: "output/genealogy/screenshots/hybrid-individual.png", fullPage: true });
    // Ordinary people expose only two ancestor and five descendant generations.
    const overview = await page.evaluate(() => api("overview"));
    let ordinary;
    for (const place of overview.settlements) {
      const atlas = await page.evaluate((id) => api(`resident-atlas?place=${id}&limit=20`), place.id);
      ordinary = atlas.people.find((p) => Number(p.id.split(":")[2]) > 0);
      if (ordinary) break;
    }
    assert.ok(ordinary);
    await page.fill("#identity", ordinary.id);
    await page.locator("#search button.primary").click();
    await page.waitForFunction(() => document.querySelector("#depth").dataset.maxDepth === "5");
    await page.selectOption("#direction", "ancestors");
    await page.waitForFunction(() => document.querySelector("#depth").dataset.maxDepth === "2");
    assert.ok(Number(await page.locator("#depth").inputValue()) <= 2);
    await page.selectOption("#depth", "1");
    await page.waitForFunction(() => !document.querySelector("#treeNote").textContent.includes("Loading"));
    assert.equal(await page.locator("#error").isVisible(), false);
    await page.screenshot({ path: "output/genealogy/screenshots/hybrid-family.png", fullPage: true });
    // Switch back to the historical mode through the same form.
    await page.click("#tabConfig");
    await page.selectOption("#populationModel", "full");
    await page.click("#generate");
    await page.waitForFunction(() => /^\d+$/.test(document.querySelector("#identity").value),
      { timeout: 60000 });
    assert.deepEqual(errors, []);
    console.log("Hybrid/full generation, opaque identities, lineage and UI screenshots passed.");
  } finally {
    await browser.close();
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
