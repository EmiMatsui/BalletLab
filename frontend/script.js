document.addEventListener("DOMContentLoaded", () => {
  const analyzeBtn = document.getElementById("analyzeBtn");
  const status = document.getElementById("status");

  // ==== ドラッグ&ドロップ処理 ====
  const dropZones = document.querySelectorAll(".drop-zone");
  dropZones.forEach(zone => {
    const input = zone.querySelector("input");

    zone.addEventListener("click", () => input.click());
    zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("hover"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("hover"));
    zone.addEventListener("drop", e => {
      e.preventDefault();
      zone.classList.remove("hover");
      if (e.dataTransfer.files.length > 0) {
        input.files = e.dataTransfer.files;
        zone.querySelector("small").textContent = `選択中: ${e.dataTransfer.files[0].name}`;
      }
    });
    input.addEventListener("change", () => {
      if (input.files.length > 0)
        zone.querySelector("small").textContent = `選択中: ${input.files[0].name}`;
    });
  });

  // ==== API呼び出し ====
  if (analyzeBtn) {
    analyzeBtn.addEventListener("click", async () => {
      const ideal = document.getElementById("ideal").files[0];
      const user = document.getElementById("user").files[0];
      if (!ideal || !user) {
        alert("2つの動画を選択してください。");
        return;
      }

      const formData = new FormData();
      formData.append("ideal_video", ideal);
      formData.append("user_video", user);

      status.textContent = "⏳ 解析中です… しばらくお待ちください";
      analyzeBtn.disabled = true;

      try {
        // 🎯 スコア
        const scoreRes = await fetch("/analyze/score", { method: "POST", body: formData });
        const scoreData = await scoreRes.json();

        // 🩰 基本フィードバック
        const basicRes = await fetch("/analyze/feedback/basic", { method: "POST", body: formData });
        const basicData = await basicRes.json();

        // 💬 ChatGPTフィードバック
        const chatRes = await fetch("/analyze/feedback/chatgpt", { method: "POST", body: formData });
        const chatData = await chatRes.json();

        const finalResult = {
          score: scoreData.score,
          feedback: {
            basic: basicData.basic_feedback,
            chatgpt: chatData.chatgpt_feedback,
          },
        };

        localStorage.setItem("ballet_result_json", JSON.stringify(finalResult));
        window.location.href = "result.html";

      } catch (err) {
        status.textContent = "⚠️ エラーが発生しました: " + err.message;
      } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = "解析開始";
      }
    });
  }

  // ==== 結果ページ処理 ====
  if (window.location.pathname.endsWith("result.html")) {
    const data = JSON.parse(localStorage.getItem("ballet_result_json") || "{}");
    const resultsDiv = document.getElementById("results");
    if (!data.score) {
      resultsDiv.innerHTML = "<p>解析結果が見つかりません。</p>";
      return;
    }

    resultsDiv.innerHTML = `
      <h2>💯 スコア: ${data.score.toFixed(1)}</h2>
      <div class="feedback">
        <h3>🩰 基本フィードバック</h3>
        <p>${data.feedback.basic.join("<br>")}</p>
        <h3>💬 ChatGPT コメント</h3>
        <p>${data.feedback.chatgpt}</p>
      </div>
    `;
  }
});
