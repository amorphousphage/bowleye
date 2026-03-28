const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const deviceSelect = document.getElementById("deviceSelect");
const modeSelect = document.getElementById("modeSelect");
const loadImageBtn = document.getElementById("loadImageBtn");
const resetBtn = document.getElementById("resetBtn");
const saveBtn = document.getElementById("saveBtn");
const output = document.getElementById("output");
const pinsControls = document.getElementById("pinsControls");
const pinSelect = document.getElementById("pinSelect");
const flipCheckbox = document.getElementById("flipVertical");

let img = new Image();
let imgLoaded = false;

// Calibration data
let arrowLeft = null;
let arrowRight = null;
let headpinLeft = null;
let headpinRight = null;

let arrowLines = [];          // TOP/BOTTOM ARROWS (restored)
let detectionBounds = [];
let pinCoordinates = {};

// ----------------------------
// Canvas resize
// ----------------------------
function resizeCanvasToImage() {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
}

// ----------------------------
// Draw everything
// ----------------------------
function draw() {
    if (!imgLoaded) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (flipCheckbox.checked) {
        ctx.save();
        ctx.scale(1, -1);
        ctx.drawImage(img, 0, -canvas.height, canvas.width, canvas.height);
        ctx.restore();
    } else {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    }

    // Draw arrow lines (TOP/BOTTOM)
    ctx.lineWidth = 2;
    ctx.strokeStyle = "yellow";
    arrowLines.forEach(line => {
        ctx.beginPath();
        ctx.moveTo(0, line.y);
        ctx.lineTo(canvas.width, line.y);
        ctx.stroke();
    });

    // Draw calibration points
    ctx.fillStyle = "yellow";
    ctx.strokeStyle = "black";
    ctx.font = "14px sans-serif";

    function drawPoint(pt, label) {
        if (!pt) return;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 6, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
        ctx.fillText(label, pt.x + 8, pt.y - 8);
    }

    drawPoint(arrowLeft, "arrow_left");
    drawPoint(arrowRight, "arrow_right");
    drawPoint(headpinLeft, "headpin_left");
    drawPoint(headpinRight, "headpin_right");

    // Draw detection trapezoid
    if (detectionBounds.length === 4) {
        ctx.strokeStyle = "red";
        ctx.beginPath();
        ctx.moveTo(detectionBounds[0].x, detectionBounds[0].y);
        for (let i = 1; i < 4; i++) {
            ctx.lineTo(detectionBounds[i].x, detectionBounds[i].y);
        }
        ctx.closePath();
        ctx.stroke();
    }

    // Draw pin coordinates
    ctx.fillStyle = "cyan";
    ctx.strokeStyle = "black";
    for (const pin in pinCoordinates) {
        const p = pinCoordinates[pin];
        ctx.beginPath();
        ctx.arc(p.x, p.y, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
        ctx.fillText(pin, p.x + 8, p.y - 8);
    }
}

// ----------------------------
// Handle clicks
// ----------------------------
function canvasClickHandler(evt) {
    if (!imgLoaded) return;

    const rect = canvas.getBoundingClientRect();
    const displayX = evt.clientX - rect.left;
    const displayY = evt.clientY - rect.top;

    const scaleX = img.naturalWidth / canvas.width;
    const scaleY = img.naturalHeight / canvas.height;

    let x = Math.round(displayX * scaleX);
    let y = Math.round(displayY * scaleY);

    if (flipCheckbox.checked) {
        y = img.naturalHeight - y;
    }

    const mode = modeSelect.value;

    // ----------------------------
    // TOP/BOTTOM ARROW LINES
    // ----------------------------
    if (mode === "arrows") {
        if (arrowLines.length < 2) {
            arrowLines.push({ y });
        } else {
            const idx = Math.abs(arrowLines[0].y - y) < Math.abs(arrowLines[1].y - y) ? 0 : 1;
            arrowLines[idx].y = y;
        }
    }

    // ----------------------------
    // 4 calibration points
    // ----------------------------
    else if (mode === "calibration_points") {
        if (!arrowLeft) arrowLeft = { x, y };
        else if (!arrowRight) arrowRight = { x, y };
        else if (!headpinLeft) headpinLeft = { x, y };
        else if (!headpinRight) headpinRight = { x, y };
        else {
            const pts = [arrowLeft, arrowRight, headpinLeft, headpinRight];
            let nearest = 0;
            let best = Infinity;
            pts.forEach((p, i) => {
                const d = (p.x - x) ** 2 + (p.y - y) ** 2;
                if (d < best) { best = d; nearest = i; }
            });
            if (nearest === 0) arrowLeft = { x, y };
            if (nearest === 1) arrowRight = { x, y };
            if (nearest === 2) headpinLeft = { x, y };
            if (nearest === 3) headpinRight = { x, y };
        }
    }

    // ----------------------------
    // Detection bounds
    // ----------------------------
    else if (mode === "detection_bounds") {
        if (detectionBounds.length < 4) {
            detectionBounds.push({ x, y });
        } else {
            let nearestIdx = 0;
            let bestDist = Infinity;
            detectionBounds.forEach((p, i) => {
                const d = (p.x - x) ** 2 + (p.y - y) ** 2;
                if (d < bestDist) { bestDist = d; nearestIdx = i; }
            });
            detectionBounds[nearestIdx] = { x, y };
        }
    }

    // ----------------------------
    // Pins
    // ----------------------------
    else if (mode === "pins") {
        const pin = pinSelect.value;
        pinCoordinates[pin] = { x, y };

        const next = (parseInt(pin) + 1).toString();
        if (next <= "10") pinSelect.value = next;
    }

    draw();
}

canvas.addEventListener("click", canvasClickHandler);

// ----------------------------
// Load image
// ----------------------------
loadImageBtn.addEventListener("click", () => {
    const device = deviceSelect.value;
    const url = `/calibration/api/frame?device=${encodeURIComponent(device)}`;

    imgLoaded = false;
    img = new Image();

    img.onload = () => {
        imgLoaded = true;
        resizeCanvasToImage();
        draw();
    };

    img.onerror = () => alert("Failed to load image");

    img.src = url + "&_ts=" + Date.now();

});

// ----------------------------
// Mode change
// ----------------------------
modeSelect.addEventListener("change", () => {
    pinsControls.style.display = (modeSelect.value === "pins") ? "block" : "none";
});

// ----------------------------
// Reset
// ----------------------------
resetBtn.addEventListener("click", () => {
    const mode = modeSelect.value;

    if (mode === "arrows") {
        arrowLines = [];
    } else if (mode === "calibration_points") {
        arrowLeft = arrowRight = headpinLeft = headpinRight = null;
    } else if (mode === "detection_bounds") {
        detectionBounds = [];
    } else if (mode === "pins") {
        pinCoordinates = {};
    }

    draw();
});

// ----------------------------
// Save
// ----------------------------
saveBtn.addEventListener("click", async () => {

    // Compute arrow line min/max
    let max_y_arrows_coordinate = null;
    let min_y_arrows_coordinate = null;

    if (arrowLines.length === 2) {
        const y1 = arrowLines[0].y;
        const y2 = arrowLines[1].y;
        max_y_arrows_coordinate = Math.max(y1, y2);
        min_y_arrows_coordinate = Math.min(y1, y2);
    }

    // Format output exactly as requested
    const payload = {
        detection_bounds: [detectionBounds.map(p => [p.x, p.y])],
                         arrow_left: arrowLeft ? [arrowLeft.x, arrowLeft.y] : null,
                         arrow_right: arrowRight ? [arrowRight.x, arrowRight.y] : null,
                         headpin_left: headpinLeft ? [headpinLeft.x, headpinLeft.y] : null,
                         headpin_right: headpinRight ? [headpinRight.x, headpinRight.y] : null,
                         max_y_arrows_coordinate,
                         min_y_arrows_coordinate
    };

    // Show formatted output in the UI
    output.textContent =
    `detection_bounds = ${JSON.stringify(payload.detection_bounds)}\n` +
    `arrow_left = ${JSON.stringify(payload.arrow_left)}\n` +
    `arrow_right = ${JSON.stringify(payload.arrow_right)}\n` +
    `headpin_left = ${JSON.stringify(payload.headpin_left)}\n` +
    `headpin_right = ${JSON.stringify(payload.headpin_right)}\n` +
    `max_y_arrows_coordinate = ${payload.max_y_arrows_coordinate}\n` +
    `min_y_arrows_coordinate = ${payload.min_y_arrows_coordinate}\n`;

    try {
        const res = await fetch("/calibration/api/save_calibration", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        console.log("Saved:", await res.json());
    } catch (e) {
        console.error("Save failed", e);
    }
});

