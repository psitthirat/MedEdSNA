// Share row + GoatCounter view count, wired the same way on the hub page
// and every field dashboard. Self-contained (no dependency on app.js/DATA)
// so the hub page -- which never loads a field's data.js -- can use it too.
(function () {
  "use strict";

  // Site code from https://www.goatcounter.com -- must match the
  // data-goatcounter URL in every dashboard/*/index.html page.
  const GOATCOUNTER_CODE = "psitthirat";

  function wireShareAndViews() {
    const copyBtn = document.getElementById("share-copy");
    const shareX = document.getElementById("share-x");
    const shareLinkedin = document.getElementById("share-linkedin");
    const shareEmail = document.getElementById("share-email");
    const viewCount = document.getElementById("view-count");
    if (!copyBtn && !shareX && !shareLinkedin && !shareEmail && !viewCount) return;

    const pageUrl = window.location.href;
    const shareText = document.title;

    if (shareX) {
      shareX.href = `https://twitter.com/intent/tweet?url=${encodeURIComponent(pageUrl)}&text=${encodeURIComponent(shareText)}`;
    }
    if (shareLinkedin) {
      shareLinkedin.href = `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(pageUrl)}`;
    }
    if (shareEmail) {
      shareEmail.href = `mailto:?subject=${encodeURIComponent(shareText)}&body=${encodeURIComponent(pageUrl)}`;
    }
    if (copyBtn) {
      copyBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(pageUrl).then(() => {
          const original = copyBtn.textContent;
          copyBtn.textContent = "✓";
          copyBtn.classList.add("copied");
          setTimeout(() => {
            copyBtn.textContent = original;
            copyBtn.classList.remove("copied");
          }, 1500);
        });
      });
    }
    if (viewCount) {
      fetch(`https://${GOATCOUNTER_CODE}.goatcounter.com/counter/TOTAL.json`)
        .then((r) => r.json())
        .then((d) => { viewCount.textContent = Number(d.count).toLocaleString(); })
        .catch(() => {});
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", wireShareAndViews);
  else wireShareAndViews();
})();
