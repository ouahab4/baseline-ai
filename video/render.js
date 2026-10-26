// Render film.html frame by frame into an MP4 (video only; audio is muxed by build.sh).
// Usage: node render.js <url-of-film.html> <out.mp4> [fps]
//        node render.js <url> --stills <dir>   (a few key frames per scene, for review)
const { chromium } = require("playwright");
const { spawn } = require("child_process");
const fs = require("fs");

(async () => {
  const [url, out, arg3] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  page.on("pageerror", e => console.error("page error:", e));
  await page.goto(url, { waitUntil: "networkidle" });
  const total = await page.evaluate(() => window.ready);
  const canvas = await page.$("canvas");

  if (out === "--stills") {
    fs.mkdirSync(arg3, { recursive: true });
    const scenes = await page.evaluate(() => TL.scenes.map(s => [s.id, s.start, s.duration]));
    for (const [id, start, dur] of scenes) {
      for (const f of [0.35, 0.92]) {
        const t = start + dur * f;
        await page.evaluate(t => window.renderAt(t), t);
        await canvas.screenshot({ path: `${arg3}/${id}-${Math.round(f * 100)}.png` });
      }
    }
    await browser.close();
    return;
  }

  const fps = Number(arg3 || 30);
  const frames = Math.ceil(total * fps);
  const ff = spawn("ffmpeg", ["-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", String(fps), "-i", "-",
    "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", out], { stdio: ["pipe", "inherit", "inherit"] });
  const t0 = Date.now();
  for (let i = 0; i < frames; i++) {
    await page.evaluate(t => window.renderAt(t), i / fps);
    const buf = await canvas.screenshot({ type: "jpeg", quality: 95 });
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once("drain", r));
    if (i % 300 === 0) console.log(`frame ${i}/${frames} (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
  }
  ff.stdin.end();
  await new Promise(r => ff.on("close", r));
  await browser.close();
  console.log(`done: ${frames} frames in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
})();
