/**
 * Discrete-step animation controls: 播放 / 暂停 / 上一步 / 下一步 / 重置.
 *
 * Usage:
 *   const player = AnimStepPlayer({
 *     root: document.getElementById("anim-cause"),  // contains buttons
 *     steps: [fn0, fn1, ...] or [{apply, cap}, ...],
 *     apply(step, index, total),   // optional if steps are functions
 *     intervalMs: 1100,
 *   });
 *   player.go(0);
 *
 * Buttons (inside root):
 *   [data-act="play"] [data-act="pause"] [data-act="prev"]
 *   [data-act="next"] [data-act="reset"]
 * Optional: [data-step] shows "i / n"
 */
(function (global) {
  function AnimStepPlayer(opts) {
    const root = opts.root;
    const steps = opts.steps;
    const intervalMs = opts.intervalMs || 1100;
    const onApply =
      opts.apply ||
      function (step, i, n) {
        if (typeof step === "function") step(i, n);
        else if (step && typeof step.apply === "function") step.apply(i, n);
      };
    let i = 0;
    let timer = null;
    const btn = (act) => root.querySelector('[data-act="' + act + '"]');
    const stepLabel = root.querySelector("[data-step]");

    function total() {
      return steps.length;
    }

    function render() {
      onApply(steps[i], i, total());
      if (stepLabel) stepLabel.textContent = i + 1 + " / " + total();
      const playing = timer != null;
      const play = btn("play");
      const pause = btn("pause");
      if (play) play.disabled = playing || i >= total() - 1;
      if (pause) pause.disabled = !playing;
      const prev = btn("prev");
      const next = btn("next");
      if (prev) prev.disabled = playing || i <= 0;
      if (next) next.disabled = playing || i >= total() - 1;
    }

    function go(n) {
      i = Math.max(0, Math.min(total() - 1, n));
      render();
    }

    function pause() {
      if (timer != null) {
        clearInterval(timer);
        timer = null;
      }
      render();
    }

    function play() {
      if (typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches) {
        go(total() - 1);
        return;
      }
      pause();
      if (i >= total() - 1) i = 0;
      render();
      timer = setInterval(function () {
        if (i >= total() - 1) {
          pause();
          return;
        }
        i += 1;
        render();
        if (i >= total() - 1) pause();
      }, intervalMs);
      render();
    }

    function next() {
      pause();
      go(i + 1);
    }

    function prev() {
      pause();
      go(i - 1);
    }

    function reset() {
      pause();
      go(0);
    }

    const playEl = btn("play");
    const pauseEl = btn("pause");
    const prevEl = btn("prev");
    const nextEl = btn("next");
    const resetEl = btn("reset");
    if (playEl) playEl.addEventListener("click", play);
    if (pauseEl) pauseEl.addEventListener("click", pause);
    if (prevEl) prevEl.addEventListener("click", prev);
    if (nextEl) nextEl.addEventListener("click", next);
    if (resetEl) resetEl.addEventListener("click", reset);

    return { go: go, play: play, pause: pause, next: next, prev: prev, reset: reset };
  }

  global.AnimStepPlayer = AnimStepPlayer;

  /** Standard control strip HTML helper (optional). */
  global.animControlsHtml = function () {
    return (
      '<span class="anim-controls">' +
      '<button type="button" class="primary" data-act="play">播放</button>' +
      '<button type="button" data-act="pause">暂停</button>' +
      '<button type="button" data-act="prev">上一步</button>' +
      '<button type="button" data-act="next">下一步</button>' +
      '<button type="button" data-act="reset">重置</button>' +
      '<span class="anim-step" data-step>1 / 1</span>' +
      "</span>"
    );
  };
})(window);
