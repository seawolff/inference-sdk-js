import type { Keypoint } from '@roboflow/inference-sdk';

export interface PoseResult {
    name: string;
    confidence: number; // fraction of geometric conditions met (0–1)
}

/**
 * Returns the fraction of applicable conditions that are true.
 * Null means the joint wasn't visible — those entries are skipped entirely
 * so a partially-visible pose isn't unfairly penalized.
 */
export function scoreConditions(conditions: Array<boolean | null>): number {
    const applicable = conditions.filter((c): c is boolean => c !== null);
    return applicable.length === 0 ? 0 : applicable.filter(Boolean).length / applicable.length;
}


/**
 * Classify which yoga pose is being held based on COCO keypoint positions.
 *
 * Thresholds are normalized by shoulder width so classification works
 * at any distance from the camera.
 *
 * Returns { name, confidence } — confidence is the fraction of geometric
 * conditions that matched. Returns "Unknown" when the best match is below 50%.
 */
export function classifyPose(keypoints: Keypoint[]): PoseResult {
    // Only use joints the model is confident about (below 0.3 = unreliable)
    const visible = (i: number): Keypoint | null => {
        const k = keypoints[i];
        return k && k.confidence >= 0.3 ? k : null;
    };

    // Pull out only the joints we actually check — indices follow COCO order
    const nose          = visible(0);
    const leftShoulder  = visible(5);
    const rightShoulder = visible(6);
    const leftWrist     = visible(9);
    const rightWrist    = visible(10);
    const leftHip       = visible(11);
    const rightHip      = visible(12);
    const leftKnee      = visible(13);
    const rightKnee     = visible(14);

    // Shoulder width is our scale reference — a larger value means the person
    // is closer to the camera, so all pixel thresholds scale with it.
    const scale = leftShoulder && rightShoulder
        ? Math.abs(leftShoulder.x - rightShoulder.x)
        : 120;

    // Note: image coordinates have y=0 at the TOP. So a joint that is physically
    // higher on the body has a SMALLER y value. "wrist above nose" = wrist.y < nose.y.

    const poses = [
        {
            name: "Mountain Pose",
            // Wrists hang below the shoulders (larger y) and stay close to the hips
            score: scoreConditions([
                leftWrist && leftShoulder   ? leftWrist.y > leftShoulder.y   : null,
                rightWrist && rightShoulder ? rightWrist.y > rightShoulder.y : null,
                leftWrist && leftHip        ? Math.abs(leftWrist.x - leftHip.x)   < scale * 0.6 : null,
                rightWrist && rightHip      ? Math.abs(rightWrist.x - rightHip.x) < scale * 0.6 : null,
            ]),
        },
        {
            name: "Warrior II",
            // Wrists are level with the shoulders (same y) and spread far apart (large x gap)
            score: scoreConditions([
                leftWrist && leftShoulder   ? Math.abs(leftWrist.y - leftShoulder.y)   < scale * 0.5 : null,
                rightWrist && rightShoulder ? Math.abs(rightWrist.y - rightShoulder.y) < scale * 0.5 : null,
                leftWrist && rightWrist     ? Math.abs(leftWrist.x - rightWrist.x)     > scale * 2   : null,
            ]),
        },
        {
            name: "Tree Pose",
            // Wrists raised above the nose (smaller y) and one knee higher than the other
            score: scoreConditions([
                nose && leftWrist     ? leftWrist.y  < nose.y : null,
                nose && rightWrist    ? rightWrist.y < nose.y : null,
                leftKnee && rightKnee ? Math.abs(leftKnee.y - rightKnee.y) > scale * 0.8 : null,
            ]),
        },
        {
            name: "Headstand",
            // Body is inverted: hips appear above the nose in the image (smaller y)
            score: scoreConditions([
                nose && leftHip  ? leftHip.y  < nose.y : null,
                nose && rightHip ? rightHip.y < nose.y : null,
            ]),
        },
    ];

    const best = [...poses].sort((a, b) => b.score - a.score)[0];
    return best.score >= 0.5
        ? { name: best.name, confidence: best.score }
        : { name: "Unknown", confidence: 0 };
}
