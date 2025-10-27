// Lightbox navigation via gamepad and wheel (JS + JSDoc). No build.

let installed = false;
/** @type {(number | undefined)[]} */
const intervals = [];

/** @param {() => boolean} pred @param {number} timeoutMs */
function sleepUntil(pred, timeoutMs) {
  return new Promise((resolve) => {
    const start = Date.now();
    const timer = window.setInterval(() => {
      if (pred() || Date.now() - start > timeoutMs) {
        window.clearInterval(timer);
        resolve(undefined);
      }
    }, 20);
  });
}

/** @param {number} dir */
function switchOffset(dir) {
  try { if (typeof modalImageSwitch === 'function') modalImageSwitch(dir); } catch {}
}

/** @param {GamepadEvent} event */
function onGamepadConnected(event) {
  if (!(event instanceof GamepadEvent)) return;
  const index = event.gamepad.index; let waiting = false;
  const id = window.setInterval(async ()=>{
    try { /* @ts-ignore */ if (!opts?.js_modal_lightbox_gamepad || waiting) return; } catch { if (waiting) return; }
    const gp = navigator.getGamepads()[index]; const x = gp?.axes?.[0] ?? 0;
    if (x <= -0.3) { switchOffset(-1); waiting = true; }
    else if (x >= 0.3) { switchOffset(1); waiting = true; }
    if (waiting) {
      let delay = 400; try { /* @ts-ignore */ delay = Number(opts?.js_modal_lightbox_gamepad_repeat ?? 400); } catch {}
      await sleepUntil(()=>{ const cur = navigator.getGamepads()[index]; const v = cur?.axes?.[0] ?? 0; return v < 0.3 && v > -0.3; }, delay);
      waiting = false;
    }
  }, 10);
  intervals[index] = id;
}

/** @param {GamepadEvent} event */
function onGamepadDisconnected(event) {
  if (!(event instanceof GamepadEvent)) return;
  const id = intervals[event.gamepad.index]; if (id !== undefined) { window.clearInterval(id); intervals[event.gamepad.index]=undefined; }
}

let wheelLock = false;
/** @param {WheelEvent} event */
function onWheel(event) {
  try { /* @ts-ignore */ if (!opts?.js_modal_lightbox_gamepad || wheelLock) return; } catch { if (wheelLock) return; }
  wheelLock = true;
  if (event.deltaX <= -0.6) switchOffset(-1); else if (event.deltaX >= 0.6) switchOffset(1);
  let delay = 400; try { /* @ts-ignore */ delay = Number(opts?.js_modal_lightbox_gamepad_repeat ?? 400); } catch {}
  window.setTimeout(()=>{ wheelLock = false; }, delay);
}

export function install(){ if (installed) return; installed=true; window.addEventListener('gamepadconnected', onGamepadConnected); window.addEventListener('gamepaddisconnected', onGamepadDisconnected); window.addEventListener('wheel', onWheel); }
