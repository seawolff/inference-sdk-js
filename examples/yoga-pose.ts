import { connectors, webrtc, streams } from '@roboflow/inference-sdk';
import type { WebRTCOutputData, PoseDetectionOutput, KeypointPrediction, Keypoint } from '@roboflow/inference-sdk';
import { classifyPose } from './yoga-pose-classifier';

// Get DOM elements
const apiKeyInput = document.getElementById("apiKeyInput") as HTMLInputElement;
const serverUrlInput = document.getElementById("serverUrlInput") as HTMLInputElement;
const startBtn = document.getElementById("startBtn") as HTMLButtonElement;
const stopBtn = document.getElementById("stopBtn") as HTMLButtonElement;
const statusEl = document.getElementById("status")!;
const videoEl = document.getElementById("video") as HTMLVideoElement;
const canvasEl = document.getElementById("overlay") as HTMLCanvasElement;
const statsEl = document.getElementById("stats")!;
const ctx = canvasEl.getContext("2d")!;

// Track active connection
let activeConnection: Awaited<ReturnType<typeof webrtc.useStream>> | null = null;

// COCO keypoint order produced by yolov8n-pose-640
const KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
];

// Pairs of keypoint indices to connect with lines when drawing the skeleton.
// Each number maps to a joint name via KEYPOINT_NAMES above.
/*
        0 (nose)
       / \
      1   2  (eyes)
      |   |
      3   4  (ears)

      5 - 6  (shoulders)
     /|   |\
    7 |   | 8  (elbows)
    | |   | |
    9 |   | 10 (wrists)
      |   |
     11 - 12 (hips)
      |   |
     13  14  (knees)
      |   |
     15  16  (ankles)
*/
const SKELETON: [number, number][] = [
    [0, 1], [0, 2],     // nose → eyes
    [1, 3], [2, 4],     // eyes → ears
    [5, 6],             // shoulder to shoulder
    [5, 7], [7, 9],     // left shoulder → elbow → wrist
    [6, 8], [8, 10],    // right shoulder → elbow → wrist
    [5, 11], [6, 12],   // shoulders → hips
    [11, 12],           // hip to hip
    [11, 13], [13, 15], // left hip → knee → ankle
    [12, 14], [14, 16], // right hip → knee → ankle
];

// Pose estimation workflow using yolov8n-pose-640
const WORKFLOW_SPEC = {
    "version": "1.0",
    "inputs": [
        { "type": "InferenceImage", "name": "image" }
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_keypoint_detection_model@v2",
            "name": "pose",
            "images": "$inputs.image",
            "model_id": "yolov8n-pose-640"
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.pose.predictions"
        }
    ]
};

/**
 * Update status display
 */
function setStatus(text: string) {
    statusEl.textContent = text;
    console.log("[UI Status]", text);
}

/**
 * Draw keypoints and skeleton for all detected people onto the canvas overlay,
 * then classify and display the current pose.
 */
function drawPoseOverlay(output: { predictions: PoseDetectionOutput }) {
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

    const poseOutput = output.predictions;
    const predictions = poseOutput?.predictions;
    if (!predictions || predictions.length === 0) {
        statsEl.textContent = "No pose detected";
        return;
    }

    // The model reports coordinates in the source frame resolution (e.g. 640×480).
    // Scale them to match however large the canvas is actually rendered on screen.
    const scaleX = canvasEl.width / (poseOutput.image?.width ?? canvasEl.width);
    const scaleY = canvasEl.height / (poseOutput.image?.height ?? canvasEl.height);

    predictions.forEach((person: KeypointPrediction) => {
        const keypoints = person.keypoints;
        if (!keypoints) return;

        // Draw skeleton lines
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = "rgba(99, 236, 130, 0.85)";
        SKELETON.forEach(([from, to]) => {
            const a = keypoints[from];
            const b = keypoints[to];
            if (!a || !b || a.confidence < 0.3 || b.confidence < 0.3) return;
            ctx.beginPath();
            ctx.moveTo(a.x * scaleX, a.y * scaleY);
            ctx.lineTo(b.x * scaleX, b.y * scaleY);
            ctx.stroke();
        });

        // Draw keypoint dots
        keypoints.forEach((kp: Keypoint) => {
            if (kp.confidence < 0.3) return;
            ctx.fillStyle = "rgba(255, 80, 80, 0.9)";
            ctx.beginPath();
            ctx.arc(kp.x * scaleX, kp.y * scaleY, 5, 0, 2 * Math.PI);
            ctx.fill();
        });
    });

    const person = predictions[0];
    const keypoints = person.keypoints ?? [];
    const pose = classifyPose(keypoints);
    const visibleCount = keypoints.filter((kp: Keypoint) => kp.confidence >= 0.3).length;

    // Draw pose label on canvas — painted twice to create a readable drop shadow:
    // dark version offset by 1px, then the coloured version on top.
    const label = pose.name === "Unknown"
        ? "Hold a pose..."
        : `${pose.name}  ${(pose.confidence * 100).toFixed(0)}%`;
    ctx.font = "bold 22px system-ui, sans-serif";
    ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
    ctx.fillText(label, 14, 36);
    ctx.fillStyle = pose.name === "Unknown" ? "rgba(255,255,255,0.7)" : "rgba(99, 236, 130, 1)";
    ctx.fillText(label, 13, 35);

    statsEl.textContent = [
        `Pose: ${pose.name}`,
        `Person confidence: ${(person.confidence * 100).toFixed(0)}%`,
        `Visible joints: ${visibleCount} / ${KEYPOINT_NAMES.length}`,
    ].join("  |  ");
}

/**
 * Handle incoming WebRTC inference data
 */
function handleData(data: WebRTCOutputData) {
    if (data.errors.length > 0) {
        console.warn("[UI] Inference errors:", data.errors);
        return;
    }
    // Cast at the SDK boundary — onData receives Record<string, any> until
    // WebRTCOutputData's generic T is threaded through useStream's param types.
    const output = data.serialized_output_data as { predictions: PoseDetectionOutput } | null;
    if (!output) return;
    drawPoseOverlay(output);
}

/**
 * Establish WebRTC connection to Roboflow pose estimation pipeline
 */
async function connectToRoboflow() {
    const apiKey = apiKeyInput.value.trim();
    const serverUrl = serverUrlInput.value.trim();

    if (!apiKey) {
        throw new Error("API key is required");
    }

    const source = await streams.useCamera({
        video: {
            facingMode: { ideal: "environment" },
            width: { ideal: 640 },
            height: { ideal: 480 },
            frameRate: { ideal: 30, max: 30 }
        },
        audio: false
    });

    // Show local camera feed — keypoints are drawn by drawPoseOverlay, not Roboflow
    videoEl.srcObject = source;
    videoEl.onloadedmetadata = () => {
        canvasEl.width = videoEl.videoWidth;
        canvasEl.height = videoEl.videoHeight;
    };

    const connection = await webrtc.useStream({
        source,
        connector: connectors.withApiKey(apiKey, { serverUrl }),
        wrtcParams: {
            workflowSpec: WORKFLOW_SPEC,
            imageInputName: "image",
            // Receive raw prediction data only — no annotated video stream from Roboflow.
            // We draw our own skeleton overlay on the local camera feed instead.
            dataOutputNames: ["predictions"],
            streamOutputNames: [],
        },
        onData: handleData,
        options: {
            // Send full camera resolution to the model rather than letting the
            // browser downscale it, which improves small-joint detection accuracy.
            disableInputStreamDownscaling: true
        }
    });

    return connection;
}

/**
 * Start camera and begin pose detection
 */
async function start() {
    if (activeConnection) {
        console.warn("[UI] Already connected");
        return;
    }

    startBtn.disabled = true;
    setStatus("Connecting...");

    try {
        activeConnection = await connectToRoboflow();
        setStatus("Connected - detecting poses");
        stopBtn.disabled = false;
        console.log("[UI] Successfully connected!");
    } catch (err) {
        console.error("[UI] Connection failed:", err);

        if ((err as Error).message === "API key is required") {
            alert("Please enter your Roboflow API key!");
            apiKeyInput.focus();
        }

        setStatus(`Error: ${(err as Error).message}`);
        startBtn.disabled = false;
        activeConnection = null;
    }
}

/**
 * Stop pose detection and release camera
 */
async function stop() {
    if (!activeConnection) return;

    stopBtn.disabled = true;
    setStatus("Stopping...");

    try {
        await activeConnection.cleanup();
        console.log("[UI] Cleanup complete");
    } catch (err) {
        console.error("[UI] Cleanup error:", err);
    } finally {
        activeConnection = null;

        const stream = videoEl.srcObject as MediaStream | null;
        if (stream) streams.stopStream(stream);
        videoEl.srcObject = null;

        ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
        statsEl.textContent = "No pose detected";
        startBtn.disabled = false;
        stopBtn.disabled = true;
        setStatus("Idle");
    }
}

// Attach event listeners
startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);

// Cleanup on page unload
window.addEventListener("pagehide", () => {
    if (activeConnection) activeConnection.cleanup();
});

window.addEventListener("beforeunload", () => {
    if (activeConnection) activeConnection.cleanup();
});
