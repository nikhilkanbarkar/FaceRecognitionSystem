const peopleList = document.getElementById("peopleList");
const peopleCount = document.getElementById("peopleCount");
const facesCount = document.getElementById("facesCount");
const recognisedCount = document.getElementById("recognisedCount");
const unknownCount = document.getElementById("unknownCount");
const fps = document.getElementById("fps");
const uptime = document.getElementById("uptime");
const systemStatus = document.getElementById("systemStatus");
const feedState = document.getElementById("feedState");
const errorBox = document.getElementById("errorBox");

let lastPeopleSignature = "";

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function formatUptime(seconds) {
    seconds = Math.floor(Number(seconds) || 0);

    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;

    if (h > 0) return `${h}h ${m}m`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
}

function personCard(person) {
    const isUnknown = person.recognised_name === "UNKNOWN";
    const image = person.has_image
        ? `<img src="/person_image/${encodeURIComponent(person.recognised_name)}"
                alt="${escapeHtml(person.name)}"
                onerror="this.parentElement.classList.add('image-failed'); this.remove();">`
        : `<div class="avatar-placeholder">${isUnknown ? "?" : escapeHtml(person.name.slice(0, 1).toUpperCase())}</div>`;

    const statusClass = String(person.status || "").toLowerCase().replaceAll(" ", "-");

    return `
        <article class="person-card ${isUnknown ? "unknown-card" : ""}">
            <div class="person-top">
                <div class="avatar">${image}</div>
                <div class="person-title">
                    <span class="track-label">TRACK ${escapeHtml(person.track_id)}</span>
                    <h3>${escapeHtml(person.name)}</h3>
                    <span class="similarity">${escapeHtml(person.similarity_percent)}% match</span>
                </div>
            </div>

            <div class="person-details">
                <div class="detail">
                    <span>RELATIONSHIP</span>
                    <strong>${escapeHtml(person.relation)}</strong>
                </div>
                <div class="detail">
                    <span>STATUS</span>
                    <strong class="status-text ${escapeHtml(statusClass)}">${escapeHtml(person.status)}</strong>
                </div>
                <div class="detail full">
                    <span>DESCRIPTION</span>
                    <p>${escapeHtml(person.description)}</p>
                </div>
            </div>
        </article>
    `;
}

function updatePeople(people) {
    const signature = JSON.stringify(
        people.map(p => [
            p.track_id,
            p.recognised_name,
            p.similarity_percent,
            p.relation,
            p.status,
            p.description,
            p.has_image
        ])
    );

    if (signature === lastPeopleSignature) return;
    lastPeopleSignature = signature;

    if (!people.length) {
        peopleList.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">◎</div>
                <h3>No faces detected</h3>
                <p>Stand in front of the camera to begin recognition.</p>
            </div>
        `;
        return;
    }

    peopleList.innerHTML = people
        .sort((a, b) => a.track_id - b.track_id)
        .map(personCard)
        .join("");
}

async function refreshStatus() {
    try {
        const response = await fetch("/api/status", { cache: "no-store" });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();

        facesCount.textContent = data.faces_detected ?? 0;
        recognisedCount.textContent = data.recognised ?? 0;
        unknownCount.textContent = data.unknown ?? 0;
        fps.textContent = Number(data.fps ?? 0).toFixed(1);
        uptime.textContent = formatUptime(data.uptime);

        const people = data.people || [];
        peopleCount.textContent = people.length;
        updatePeople(people);

        if (data.running) {
            systemStatus.innerHTML = `
                <span class="status-dot online"></span>
                <span>RUNNING</span>
            `;
            feedState.textContent = "LIVE · RECOGNITION ACTIVE";
        } else {
            systemStatus.innerHTML = `
                <span class="status-dot offline"></span>
                <span>STOPPED</span>
            `;
            feedState.textContent = "CAMERA STOPPED";
        }

        if (data.error) {
            errorBox.textContent = data.error;
            errorBox.classList.remove("hidden");
        } else {
            errorBox.classList.add("hidden");
        }

    } catch (error) {
        systemStatus.innerHTML = `
            <span class="status-dot offline"></span>
            <span>BACKEND ERROR</span>
        `;
        errorBox.textContent = `Unable to read backend status: ${error.message}`;
        errorBox.classList.remove("hidden");
    }
}

refreshStatus();
setInterval(refreshStatus, 350);
