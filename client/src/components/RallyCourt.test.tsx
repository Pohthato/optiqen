import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { RallyCourt, RallyView, VIEWS, project, region } from "./RallyCourt";

const RALLY: RallyView = {
  shots: [
    {
      shot: 1, hitter: "near", hitMs: 500, name: { said: "serve" },
      path: [[500, 2.0, 4.5, 1.1], [700, 2.6, 8.0, 2.0], [900, 3.0, 11.0, 1.4]],
      landing: null,
      players: { near: { point: [2.0, 4.2], covariance: [[0.04, 0], [0, 0.09]] }, far: { point: [3.0, 11.5], covariance: [[0.05, 0], [0, 0.1]] } },
    },
    {
      shot: 2, hitter: "far", hitMs: 1000, name: { said: "cross smash" },
      path: [[1000, 3.0, 11.0, 2.4], [1200, 1.5, 7.0, 1.7], [1350, 0.2, 13.6, 0.0]],
      landing: { point: [0.2, 13.6], covariance: [[0.01, 0], [0, 0.04]], call: "out", pIn: 0.01 },
      players: null,
    },
  ],
  findings: [
    { id: "F1", kind: "end", shots: [2], evidence: [{ frame: 31, timeMs: 1033.3 }], text: "Shot 2, the far player's, came down out (99%), so the near player won the point." },
    { id: "F2", kind: "shot", shots: [1], evidence: [{ frame: 16, timeMs: 533.3 }], text: "Shot 1, the near player's: serve." },
  ],
};

describe("the rally on the court", () => {
  it("puts the court's centre in the middle of every view, the far end further up from behind, and up as up", () => {
    for (const view of Object.values(VIEWS)) expect(project([2.59, 6.7, 0], view)).toEqual([320, 210]);
    const near = project([2.59, 0, 0], VIEWS.behind)!;
    const far = project([2.59, 13.4, 0], VIEWS.behind)!;
    expect(far[1]).toBeLessThan(near[1]);
    expect(project([2.59, 6.7, 1.55], VIEWS.behind)![1]).toBeLessThan(210);
    const left = project([0, 6.7, 0], VIEWS.above)!;
    const right = project([5.18, 6.7, 0], VIEWS.above)!;
    expect(left[0]).toBeLessThan(right[0]);
  });

  it("draws a landing's 95 % region as wide as its covariance says", () => {
    const outline = region({ point: [1, 2], covariance: [[0.04, 0], [0, 0.09]] });
    const xs = outline.map(([x]) => x);
    const ys = outline.map(([, y]) => y);
    expect(Math.max(...xs) - 1).toBeCloseTo(Math.sqrt(5.991) * 0.2, 3);
    expect(Math.max(...ys) - 2).toBeCloseTo(Math.sqrt(5.991) * 0.3, 3);
    expect(outline.every(([, , z]) => z === 0)).toBe(true);
  });

  it("draws every shot, the landing with its call and the findings to choose from, and nothing else", () => {
    const markup = renderToStaticMarkup(<RallyCourt rally={RALLY} />);
    expect(markup).toContain('aria-label="2 shots drawn on the court"');
    expect(markup.match(/data-shot="/g)).toHaveLength(2);
    expect(markup).toContain("Landing: out");
    expect(markup).toContain("Shot 2: cross smash");
    expect(markup).toContain("came down out (99%)");
    expect(markup).not.toContain("Where the");
  });
});
