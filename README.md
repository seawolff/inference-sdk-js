# @roboflow/inference-sdk

Lightweight JS client for Roboflow's hosted inference API with WebRTC streaming support for real-time computer vision in the browser.

---

> **This fork includes a real-time yoga pose detection demo** built on the SDK's WebRTC streaming pipeline.
>
> **[View the demo →](examples/yoga-pose.ts)** · **[Typed prediction interfaces PR →](https://github.com/roboflow/inference-sdk-js/pulls)**
>
> **How it works:** Uses `yolov8n-pose-640` to stream body keypoints via WebRTC. Each frame, joint positions are compared geometrically — thresholds are normalized by shoulder width so classification works at any distance from the camera. No custom model training required.
>
> **Currently Supported poses:** 
> + Mountain Pose
> + Warrior II
> + Tree Pose
> + Headstand
> 
> These four were added as examples. New poses can be added easily by appending an entry to the `poses` array in [`examples/yoga-pose-classifier.ts`](examples/yoga-pose-classifier.ts) — each pose is a name and a list of geometric conditions on joint positions.
>
> Joints are referenced by their [COCO keypoint index](https://docs.ultralytics.com/datasets/pose/coco/) (e.g. `5` = left shoulder, `9` = left wrist). Coordinates have `y=0` at the top of the image, so a joint that is physically higher has a *smaller* `y` value. Conditions return `null` when a joint isn't visible, and `null` entries are skipped so partially-visible poses aren't penalized. The `scale` variable (shoulder width in pixels) is used to normalize thresholds so they work at any camera distance.
>
> ```typescript
> {
>     name: "Warrior I",
>     // Arms raised overhead: wrists above the nose
>     score: scoreConditions([
>         nose && leftWrist  ? leftWrist.y  < nose.y : null,
>         nose && rightWrist ? rightWrist.y < nose.y : null,
>     ]),
> },
> ```
>
> **Core files:**
> | File | Purpose |
> |---|---|
> | [`examples/yoga-pose.ts`](examples/yoga-pose.ts) | Main entry point — connects the webcam via `useStream()`, receives keypoint data each frame via `onData`, draws the skeleton overlay on a canvas |
> | [`examples/yoga-pose-classifier.ts`](examples/yoga-pose-classifier.ts) | Pure pose classification logic — `classifyPose(keypoints)` scores each pose against geometric conditions, `scoreConditions()` handles invisible joints gracefully |
> | [`examples/yoga-pose-classifier.test.ts`](examples/yoga-pose-classifier.test.ts) | Vitest unit tests for the classifier, one test per pose |
>
> **Key methods:**
> - `webrtc.useStream()` — establishes the WebRTC connection and streams camera frames to Roboflow
> - `onData(data)` — called each frame with raw keypoint predictions from `yolov8n-pose-640`
> - `classifyPose(keypoints)` — returns `{ name, confidence }` for the best-matching pose
> - `scoreConditions(conditions)` — returns the fraction of applicable geometric conditions that are true
>
> **To run:** `npm install && npm run dev`, then open `http://localhost:5173/examples/yoga-pose.html`

---

## Installation

```bash
npm install @roboflow/inference-sdk
```

## Quick Example

```typescript
import { useStream, connectors } from '@roboflow/inference-sdk';
import { useCamera } from '@roboflow/inference-sdk/streams';

const stream = await useCamera({ video: { facingMode: "environment" } });
const connection = await useStream({
  source: stream,
  connector: connectors.withProxyUrl('/api/init-webrtc'), // Use backend proxy
  wrtcParams: { workflowSpec: { /* ... */ } },
  onData: (data) => console.log("Inference results:", data)
});

const videoElement.srcObject = await connection.remoteStream();
```

See the [sample app](https://github.com/roboflow/inferenceSampleApp) for a complete working example.

## Error Handling

Both `connectors.withApiKey()` and `connectors.withProxyUrl()` throw a `WorkflowError` when the backend returns a structured error response (e.g. invalid workflow spec, missing model, block execution failure). For other failures (network errors, non-JSON responses) a plain `Error` is thrown. `WorkflowError` extends `Error`, so existing `catch` blocks keep working.

```typescript
import { useStream, connectors, WorkflowError } from '@roboflow/inference-sdk';

try {
  const connection = await useStream({
    source: stream,
    connector: connectors.withProxyUrl('/api/init-webrtc'),
    wrtcParams: { workflowSpec: { /* ... */ } },
    onData: (data) => console.log(data),
  });
} catch (err) {
  if (err instanceof WorkflowError) {
    // err.statusCode: HTTP status from the backend (e.g. 400)
    // err.errorData: { message, error_type, context, inner_error_type,
    //                  inner_error_message, blocks_errors }
    console.error(err.errorData.error_type, err.errorData.message);
    for (const block of err.errorData.blocks_errors ?? []) {
      console.error(`  block ${block.block_id}: ${block.property_details}`);
    }
  } else {
    console.error('Transport error:', err);
  }
}
```

For this to work with `withProxyUrl()`, your backend proxy must forward Roboflow's
error status and JSON body. The recommended pattern:

```typescript
try {
  const answer = await client.initializeWebrtcWorker({ /* ... */ });
  res.json(answer);
} catch (err) {
  if (err instanceof WorkflowError) {
    res.status(err.statusCode).json(err.errorData);
  } else {
    res.status(500).json({ message: err.message ?? 'Unknown error' });
  }
}
```

## Security Warning

**Never expose your API key in frontend code.** Always use a backend proxy for production applications. The sample app demonstrates the recommended proxy pattern.

## Get Started

For a complete working example with backend proxy setup, see:
**[github.com/roboflow/inferenceSampleApp](https://github.com/roboflow/inferenceSampleApp)**

## Resources

- [Roboflow Documentation](https://docs.roboflow.com/)
- [API Authentication Guide](https://docs.roboflow.com/api-reference/authentication)
- [Workflows Documentation](https://docs.roboflow.com/workflows)

## License

See the main repository for license information.
