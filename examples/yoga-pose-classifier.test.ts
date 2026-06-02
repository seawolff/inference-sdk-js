import { describe, it, expect } from "vitest";
import { scoreConditions, classifyPose } from "./yoga-pose-classifier";
import type { Keypoint } from "@roboflow/inference-sdk";

// Build a full 17-keypoint array with everything invisible by default.
// Override specific joints by index to set up each test scenario.
function makeKeypoints(overrides: Partial<Record<number, Partial<Keypoint>>> = {}): Keypoint[] {
    return Array.from({ length: 17 }, (_, i) => ({
        x: 0,
        y: 0,
        confidence: 0,
        class_id: i,
        class: "",
        ...overrides[i],
    }));
}

// Shorthand for a visible keypoint at a given position
function kp(x: number, y: number): Partial<Keypoint> {
    return { x, y, confidence: 0.9 };
}

// Assert that a set of keypoints is classified as the expected pose
function assertPose(keypoints: Keypoint[], expectedName: string) {
    const result = classifyPose(keypoints);
    expect(result.name).toBe(expectedName);
    expect(result.confidence).toBeGreaterThanOrEqual(0.5);
}

describe("scoreConditions", () => {
    it("returns 1 when all conditions are true", () => {
        expect(scoreConditions([true, true, true])).toBe(1);
    });

    it("returns 0 when all conditions are false", () => {
        expect(scoreConditions([false, false])).toBe(0);
    });

    it("returns the correct fraction for mixed conditions", () => {
        expect(scoreConditions([true, false, true, false])).toBe(0.5);
    });

    it("skips null entries (joint not visible)", () => {
        // Two true, one null — null is skipped so result is 2/2 not 2/3
        expect(scoreConditions([true, true, null])).toBe(1);
    });

    it("returns 0 when all entries are null", () => {
        expect(scoreConditions([null, null])).toBe(0);
    });
});

describe("classifyPose", () => {
    it("detects Mountain Pose: wrists below shoulders and close to hips", () => {
        // Shoulder width = |100 - 300| = 200px (scale reference)
        // Wrists at y=350: below shoulders (y=100) and near hips (y=300)
        assertPose(makeKeypoints({
            5:  kp(100, 100), // left shoulder
            6:  kp(300, 100), // right shoulder
            9:  kp(110, 350), // left wrist
            10: kp(290, 350), // right wrist
            11: kp(120, 300), // left hip
            12: kp(280, 300), // right hip
        }), "Mountain Pose");
    });

    it("detects Warrior II: wrists at shoulder height and spread wide", () => {
        // Shoulder width = 200px → wrist span 500 > threshold 400 (scale * 2)
        assertPose(makeKeypoints({
            5:  kp(200, 200), // left shoulder
            6:  kp(400, 200), // right shoulder
            9:  kp(50,  205), // left wrist — extended left at shoulder height
            10: kp(550, 195), // right wrist — extended right at shoulder height
        }), "Warrior II");
    });

    it("detects Tree Pose: hands above head and one knee raised", () => {
        // Wrists at y=50, above nose at y=150
        // Knee height difference = 200 > threshold 160 (scale * 0.8)
        assertPose(makeKeypoints({
            0:  kp(300, 150), // nose
            5:  kp(200, 250), // left shoulder (sets scale = 200)
            6:  kp(400, 250), // right shoulder
            9:  kp(290,  50), // left wrist — above nose
            10: kp(310,  50), // right wrist — above nose
            13: kp(250, 550), // left knee — standing leg
            14: kp(300, 350), // right knee — raised
        }), "Tree Pose");
    });

    it("detects Headstand: hips above nose in the image", () => {
        // Inverted body — nose at the bottom (large y), hips near the top (small y)
        assertPose(makeKeypoints({
            0:  kp(300, 450), // nose — at the bottom of the frame
            11: kp(250, 100), // left hip — above the nose
            12: kp(350, 100), // right hip — above the nose
        }), "Headstand");
    });

    it("returns Unknown when no joints are visible", () => {
        const result = classifyPose(makeKeypoints());
        expect(result.name).toBe("Unknown");
        expect(result.confidence).toBe(0);
    });

    it("returns Unknown when wrist span is too narrow for Warrior II", () => {
        // Wrist span = 300, scale = 200 → threshold = 400 — does not meet Warrior II
        // Wrists are below shoulders → Mountain Pose wins with partial score
        const result = classifyPose(makeKeypoints({
            5:  kp(200, 200), // left shoulder
            6:  kp(400, 200), // right shoulder
            9:  kp(150, 210), // left wrist
            10: kp(450, 210), // right wrist
        }));
        expect(result.name).not.toBe("Warrior II");
    });
});
