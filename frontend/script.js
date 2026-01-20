document.getElementById("predictBtn").addEventListener("click", predict);

async function predict() {
    const data = {
        blueFirstBlood: Number(blueFirstBlood.value),
        blueKills: Number(blueKills.value),
        blueDeaths: Number(blueDeaths.value),
        blueAssists: Number(blueAssists.value),
        blueEliteMonsters: Number(blueEliteMonsters.value),
        blueDragons: Number(blueDragons.value),
        blueHeralds: Number(blueHeralds.value),
        blueTowersDestroyed: Number(blueTowersDestroyed.value),
        blueTotalGold: Number(blueTotalGold.value),
        blueTotalExperience: Number(blueTotalExperience.value),
        blueTotalMinionsKilled: Number(blueTotalMinionsKilled.value),

        redFirstBlood: Number(redFirstBlood.value),
        redKills: Number(redKills.value),
        redDeaths: Number(redDeaths.value),
        redAssists: Number(redAssists.value),
        redEliteMonsters: Number(redEliteMonsters.value),
        redDragons: Number(redDragons.value),
        redHeralds: Number(redHeralds.value),
        redTowersDestroyed: Number(redTowersDestroyed.value),
        redTotalGold: Number(redTotalGold.value),
        redTotalExperience: Number(redTotalExperience.value),
        redTotalMinionsKilled: Number(redTotalMinionsKilled.value)
    };

    const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });

    const result = await response.json();

    const winner = result.blueWins === 1 ? "Blue Team Wins" : "Red Team Wins";
    const cls = result.blueWins === 1 ? "blue" : "red";

    const resultDiv = document.getElementById("result");

    if (result.error) {
        resultDiv.innerHTML = `<span class="error">${result.error}</span>`;
        return;
    }
    
    if (result.blueWins === 1) {
        resultDiv.innerHTML = `
            <h2 class="blue">Blue Team Wins 🟦</h2>
            <p>Win probability: ${(result.probability * 100).toFixed(2)}%</p>
        `;
    } else {
        resultDiv.innerHTML = `
            <h2 class="red">Red Team Wins 🟥</h2>
            <p>Win probability: ${(100 - result.probability * 100).toFixed(2)}%</p>
        `;
    }
}
