/** Sticky left TOC for fasterVLA daily HTML. Include at end of <body>. */
(function () {
  var main = document.querySelector("main");
  if (!main) return;
  var heads = main.querySelectorAll("h2, h3");
  if (!heads.length) return;

  function slug(text, used) {
    var base = String(text)
      .trim()
      .replace(/\s+/g, "-")
      .replace(/[^\w\u4e00-\u9fff.\-↓↑]+/g, "")
      .replace(/-+/g, "-")
      .replace(/^-|-$/g, "")
      .toLowerCase() || "sec";
    var s = base;
    var n = 2;
    while (used[s]) {
      s = base + "-" + n++;
    }
    used[s] = true;
    return s;
  }

  var used = {};
  var nav = document.createElement("nav");
  nav.className = "toc";
  nav.setAttribute("aria-label", "本页目录");
  var title = document.createElement("div");
  title.className = "toc-title";
  title.textContent = "目录";
  nav.appendChild(title);
  var list = document.createElement("div");
  list.className = "toc-list";
  nav.appendChild(list);

  var links = [];
  heads.forEach(function (h) {
    if (!h.id) h.id = slug(h.textContent, used);
    var a = document.createElement("a");
    a.href = "#" + h.id;
    a.textContent = h.textContent.trim();
    a.className = h.tagName === "H3" ? "toc-h3" : "toc-h2";
    a.dataset.target = h.id;
    list.appendChild(a);
    links.push({ a: a, el: h });
  });

  var layout = document.createElement("div");
  layout.className = "daily-layout";
  main.parentNode.insertBefore(layout, main);
  layout.appendChild(nav);
  layout.appendChild(main);

  var topBtn = document.createElement("button");
  topBtn.type = "button";
  topBtn.className = "toc-top";
  topBtn.textContent = "顶部";
  topBtn.addEventListener("click", function () {
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
  nav.appendChild(topBtn);

  function setActive() {
    var y = window.scrollY + 96;
    var cur = links[0];
    for (var i = 0; i < links.length; i++) {
      if (links[i].el.offsetTop <= y) cur = links[i];
    }
    links.forEach(function (item) {
      item.a.classList.toggle("active", item === cur);
    });
  }
  window.addEventListener("scroll", setActive, { passive: true });
  setActive();
})();
