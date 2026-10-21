// Run against a hybrid studio with births and several settlements.
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
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("response", (r) => { if (r.status() >= 400) errors.push(r.url()); });
    await page.goto(process.env.STUDIO_URL || "http://127.0.0.1:8794");
    await page.waitForFunction(() => document.querySelector("#treeNote").textContent.includes("shown"));
    await page.click("#tabIndividual");
    const overview = await page.evaluate(() => api("overview"));
    const current = await page.locator("#placePicker").inputValue();
    const other = overview.settlements.find((p) => String(p.id) !== current);
    await page.selectOption("#placePicker", String(other.id));
    await page.waitForFunction((id) => document.querySelector("#localChart").dataset.place === String(id) &&
      document.querySelector("#localChart polyline"), other.id);
    await page.click("#tabExplore");
    assert.equal(await page.locator("#chartScope").inputValue(), "place");
    assert.equal(await page.locator("#chart").getAttribute("data-place"), String(other.id));
    assert.equal(await page.locator("#chartMode").isDisabled(), true);
    await page.selectOption("#chartScope", "world");
    await page.selectOption("#chartMode", "vital");
    assert.equal(await page.locator("#chart polyline").count(), 2);
    await page.click("#tabIndividual");
    const counts = await page.evaluate(() => api("map?year=" + world.summary.end_year));
    const largest = counts.sort((a, b) => b.population - a.population)[0];
    await page.selectOption("#placePicker", String(largest.settlement));
    for (const source of ["core", "projected"]) {
      await page.selectOption("#residentSource", source);
      await page.waitForFunction((s) => atlas?.source === s, source);
      const atlas = await page.evaluate(() => ({ people: residentRows, matched: atlas.matched, groups: atlas.groups }));
      assert.ok(atlas.people.length);
      assert.equal(atlas.groups.reduce((n, g) => n + g.count, 0), atlas.matched);
      assert.ok(atlas.people.every((p) => p.source === source));
      if (source === "core") {
        await page.evaluate((id) => selectPerson(id), atlas.people[0].id);
        assert.equal(await page.locator("#personSource").getAttribute("data-source"), "core");
        assert.ok(await page.locator('.treeNode[data-source="core"]').count());
      }
    }
    // Browse an ordinary child's parents through actual mouse and keyboard actions.
    let child;
    for (const place of overview.settlements) {
      const atlas = await page.evaluate((id) => api(`resident-atlas?place=${id}&source=projected`), place.id);
      for (const candidate of atlas.people) {
        const person = await page.evaluate((id) => api("person?id=" + id), candidate.id);
        if (person.father && person.mother) { child = person; break; }
      }
      if (child) break;
    }
    assert.ok(child);
    await page.fill("#identity", child.id.split(":").slice(-2).join("."));
    await page.locator("#search button.primary").click();
    await page.waitForFunction((id) => person?.id === id &&
      document.querySelector("#treeNote").textContent.includes("shown"), child.id);
    const father = page.locator(`.treeNode[data-person="${child.father}"]`);
    await father.click();
    await page.waitForFunction((id) => person?.id === id &&
      document.querySelector("#identity").value === id, child.father);
    assert.equal(await page.locator("#personSource").getAttribute("data-source"), "projected");
    await page.click("#personBack");
    await page.waitForFunction((id) => person?.id === id &&
      document.querySelector("#treeNote").textContent.includes("shown"), child.id);
    await page.locator(`.treeNode[data-person="${child.mother}"]`).focus();
    await page.keyboard.press("Enter");
    await page.waitForFunction((id) => person?.id === id, child.mother);
    assert.equal(await page.locator("#personSource").getAttribute("data-source"), "projected");
    await page.click("#personBack");
    await page.waitForFunction((id) => person?.id === id &&
      document.querySelector("#treeNote").textContent.includes("shown"), child.id);
    fs.mkdirSync("output/genealogy/screenshots", { recursive: true });
    await page.screenshot({ path: "output/genealogy/screenshots/demography-sources.png", fullPage: true });
    assert.deepEqual(errors, []);
    console.log("City census charts, source browsing, procedural relatives and history: OK");
  } finally { await browser.close(); }
})().catch((error) => { console.error(error); process.exitCode = 1; });
