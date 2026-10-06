// tools/labeller/labeller-core.js
// Pure labelling logic for the golden-set labeller. No DOM access, so it is unit-tested
// under vitest and loaded as a classic script by index.html (works from file://).
(function (global, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  global.LabellerCore = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  // Keep these in sync with worker/geometry/court_model.py and worker/evaluation/golden.py;
  // labeller-core.test.ts fails if they drift.
  const SHOT_LABELS = ["serve", "clear", "drop", "net", "lift", "drive", "push", "smash", "block", "other"];
  const ROWS = ["back0", "long0", "short0", "short1", "long1", "back1"];
  const COLUMNS = ["dl", "sl", "c", "sr", "dr"];
  const KEYPOINT_NAMES = [
    ...ROWS.flatMap(row => COLUMNS.map(column => `${row}_${column}`)),
    "post_left_base",
    "post_left_top",
    "post_right_base",
    "post_right_top",
    "net_centre_top",
  ];
  const WINNERS = ["near", "far", "unknown"];
  const PLAYERS = ["near", "far"];
  const COCO_JOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
  ];
  // Joint pairs drawn as limbs.
  const SKELETON = [[5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [0, 1], [0, 2], [1, 3], [2, 4]];
  const MERGE_WINDOW_MS = 20;
  // Sound reaches a phone 15-45 ms after the racket meets the shuttle on a court.
  const SOUND_LEAD_MS = 30;
  const RALLY_GAP_MS = 2500;
  // A suggested hit is accepted only at a frame this close to it; anything further is a different hit.
  const SUGGESTION_WINDOW_MS = 250;
  const KEYPOINT_SOURCES = ["clicked", "adjusted", "proposed"];

  // Court geometry in metres, mirroring worker/geometry/court_model.py.
  const SINGLES_WIDTH_M = 5.18;
  const DOUBLES_MARGIN_M = 0.46;
  const COURT_LENGTH_M = 13.4;
  const NET_Y_M = COURT_LENGTH_M / 2;
  const COLUMN_X = { dl: -DOUBLES_MARGIN_M, sl: 0, c: SINGLES_WIDTH_M / 2, sr: SINGLES_WIDTH_M, dr: SINGLES_WIDTH_M + DOUBLES_MARGIN_M };
  const ROW_Y = { back0: 0, long0: 0.76, short0: NET_Y_M - 1.98, short1: NET_Y_M + 1.98, long1: COURT_LENGTH_M - 0.76, back1: COURT_LENGTH_M };
  const KEYPOINT_POSITIONS = {};
  for (const row of ROWS) for (const column of COLUMNS) KEYPOINT_POSITIONS[`${row}_${column}`] = [COLUMN_X[column], ROW_Y[row], 0];
  KEYPOINT_POSITIONS.post_left_base = [COLUMN_X.dl, NET_Y_M, 0];
  KEYPOINT_POSITIONS.post_left_top = [COLUMN_X.dl, NET_Y_M, 1.55];
  KEYPOINT_POSITIONS.post_right_base = [COLUMN_X.dr, NET_Y_M, 0];
  KEYPOINT_POSITIONS.post_right_top = [COLUMN_X.dr, NET_Y_M, 1.55];
  KEYPOINT_POSITIONS.net_centre_top = [COLUMN_X.c, NET_Y_M, 1.524];
  const FLOOR_NAMES = KEYPOINT_NAMES.filter(name => KEYPOINT_POSITIONS[name][2] === 0);
  // Painted lines as pairs of named floor points, plus the line under the net for orientation.
  const COURT_LINES = [
    ...ROWS.map(row => [`${row}_dl`, `${row}_dr`]),
    ...["dl", "sl", "sr", "dr"].map(column => [`back0_${column}`, `back1_${column}`]),
    ["back0_c", "short0_c"],
    ["short1_c", "back1_c"],
    ["post_left_base", "post_right_base"],
  ];

  const round2 = n => Math.round(n * 100) / 100;

  // A collapsed window or hidden tab gives the page zero-size layout, so coordinate maths
  // can yield NaN/Infinity. JSON.stringify would turn those into null in the export.
  function finite(value, what) {
    if (!Number.isFinite(value)) throw new Error(`${what} must be a finite number`);
    return value;
  }

  function emptyState(clipId, fps, imageSize) {
    return {
      clipId,
      fps,
      imageSize: [imageSize[0], imageSize[1]],
      keypoints: {},
      contacts: [],
      rallies: [],
      selectedPlayer: null,
      poses: {},
      hitSuggestions: [],
    };
  }

  /** Fill in fields that autosaves from older versions of the labeller lack. */
  function normalizeState(saved) {
    const state = emptyState(saved.clipId, saved.fps, saved.imageSize || [0, 0]);
    return {
      ...state,
      ...saved,
      poses: saved.poses || {},
      hitSuggestions: saved.hitSuggestions || [],
      selectedPlayer: saved.selectedPlayer || null,
    };
  }

  function addContact(state, timeMs, label) {
    const time = Math.round(finite(timeMs, "Contact time"));
    const existing = state.contacts.findIndex(contact => Math.abs(contact.timeMs - time) <= MERGE_WINDOW_MS);
    if (existing >= 0) return existing;
    state.contacts.push({ timeMs: time, label: label || "other" });
    state.contacts.sort((a, b) => a.timeMs - b.timeMs);
    return state.contacts.findIndex(contact => contact.timeMs === time);
  }

  function setLabel(state, index, label) {
    if (!SHOT_LABELS.includes(label)) throw new Error(`Unknown shot label: ${label}`);
    if (!state.contacts[index]) throw new Error(`No contact at index ${index}`);
    state.contacts[index].label = label;
  }

  function removeContact(state, index) {
    const [removed] = state.contacts.splice(index, 1);
    if (removed && state.poses) delete state.poses[String(removed.timeMs)];
  }

  function nearestContactIndex(state, timeMs, toleranceMs = Infinity) {
    let best = -1;
    let bestDistance = Infinity;
    state.contacts.forEach((contact, index) => {
      const distance = Math.abs(contact.timeMs - timeMs);
      if (distance <= toleranceMs && distance < bestDistance) {
        best = index;
        bestDistance = distance;
      }
    });
    return best;
  }

  function startRally(state, timeMs) {
    const time = Math.round(finite(timeMs, "Rally start"));
    const open = state.rallies.find(rally => rally.endMs === null);
    if (open) {
      open.startMs = time;
      return;
    }
    state.rallies.push({ startMs: time, endMs: null, winner: "unknown" });
  }

  function endRally(state, timeMs) {
    const open = state.rallies.find(rally => rally.endMs === null);
    if (!open) throw new Error("No rally is open. Press R to start one first.");
    const time = Math.round(finite(timeMs, "Rally end"));
    if (time <= open.startMs) throw new Error("A rally must end after it starts.");
    open.endMs = time;
  }

  function setWinner(state, winner) {
    if (!WINNERS.includes(winner)) throw new Error(`Unknown winner: ${winner}`);
    if (state.rallies.length === 0) throw new Error("No rally to set a winner on.");
    state.rallies[state.rallies.length - 1].winner = winner;
  }

  // timeMs is the frame the point was clicked on: handheld clips move the camera, so a
  // court point only means something together with its frame.
  // source records where the point came from: "clicked" by a person, "proposed" by the
  // geometry and accepted as-is, or "adjusted" (a proposal a person moved). Missing = clicked.
  function setKeypoint(state, name, x, y, timeMs, source) {
    if (!KEYPOINT_NAMES.includes(name)) throw new Error(`Unknown court keypoint: ${name}`);
    if (source !== undefined && !KEYPOINT_SOURCES.includes(source)) throw new Error(`Unknown keypoint source: ${source}`);
    const point = { x: round2(finite(x, "Pixel x")), y: round2(finite(y, "Pixel y")) };
    if (timeMs !== undefined) point.timeMs = Math.round(finite(timeMs, "Keypoint time"));
    if (source !== undefined) point.source = source;
    state.keypoints[name] = point;
  }

  /** A person placed or dragged a point on the frame at timeMs; proposals they move become adjusted. */
  function moveKeypoint(state, name, x, y, timeMs) {
    const existing = state.keypoints[name];
    const source = existing && (existing.source === "proposed" || existing.source === "adjusted") ? "adjusted" : "clicked";
    setKeypoint(state, name, x, y, timeMs, source);
  }

  function removeKeypoint(state, name) {
    delete state.keypoints[name];
  }

  // ---- Court completion: a homography from clicked floor points proposes the rest ----

  function solveLinear(matrix, vector) {
    const n = vector.length;
    const a = matrix.map((row, i) => [...row, vector[i]]);
    for (let col = 0; col < n; col += 1) {
      let pivot = col;
      for (let r = col + 1; r < n; r += 1) if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
      if (Math.abs(a[pivot][col]) < 1e-10) return null;
      [a[col], a[pivot]] = [a[pivot], a[col]];
      for (let r = 0; r < n; r += 1) {
        if (r === col) continue;
        const factor = a[r][col] / a[col][col];
        for (let c = col; c <= n; c += 1) a[r][c] -= factor * a[col][c];
      }
    }
    return a.map((row, i) => row[n] / row[i]);
  }

  // Similarity transform moving points to their centroid with mean distance sqrt(2), which
  // keeps the least-squares system well conditioned when mixing metres and pixels.
  function normaliser(points) {
    const cx = points.reduce((sum, p) => sum + p[0], 0) / points.length;
    const cy = points.reduce((sum, p) => sum + p[1], 0) / points.length;
    const spread = points.reduce((sum, p) => sum + Math.hypot(p[0] - cx, p[1] - cy), 0) / points.length;
    const s = spread > 0 ? Math.SQRT2 / spread : 1;
    return [[s, 0, -s * cx], [0, s, -s * cy], [0, 0, 1]];
  }

  function multiply3(a, b) {
    return a.map(row => [0, 1, 2].map(col => row[0] * b[0][col] + row[1] * b[1][col] + row[2] * b[2][col]));
  }

  function transformPoint(h, point) {
    const [x, y] = point;
    const w = h[2][0] * x + h[2][1] * y + h[2][2];
    return [(h[0][0] * x + h[0][1] * y + h[0][2]) / w, (h[1][0] * x + h[1][1] * y + h[1][2]) / w, w];
  }

  /** Least-squares homography from >= 4 point pairs; null when degenerate. */
  function homography(from, to) {
    if (from.length < 4 || from.length !== to.length) return null;
    const tf = normaliser(from);
    const tt = normaliser(to);
    const a = Array.from({ length: 8 }, () => new Array(8).fill(0));
    const b = new Array(8).fill(0);
    for (let i = 0; i < from.length; i += 1) {
      const [x, y] = transformPoint(tf, from[i]);
      const [u, v] = transformPoint(tt, to[i]);
      for (const row of [[x, y, 1, 0, 0, 0, -u * x, -u * y, u], [0, 0, 0, x, y, 1, -v * x, -v * y, v]]) {
        for (let p = 0; p < 8; p += 1) {
          b[p] += row[p] * row[8];
          for (let q = 0; q < 8; q += 1) a[p][q] += row[p] * row[q];
        }
      }
    }
    const h = solveLinear(a, b);
    if (!h) return null;
    const s = tt[0][0];
    const inverseTo = [[1 / s, 0, -tt[0][2] / s], [0, 1 / s, -tt[1][2] / s], [0, 0, 1]];
    return multiply3(multiply3(inverseTo, [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]]), tf);
  }

  function triangleArea(a, b, c) {
    return Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2;
  }

  // Points that pin down a homography. With exactly four, no three may lie on one line;
  // with more, some triangle must cover more than 0.5 square metres.
  function wellSpread(points) {
    const triples = [];
    for (let i = 0; i < points.length; i += 1)
      for (let j = i + 1; j < points.length; j += 1)
        for (let k = j + 1; k < points.length; k += 1) triples.push(triangleArea(points[i], points[j], points[k]));
    return points.length === 4 ? triples.every(area => area > 0.5) : triples.some(area => area > 0.5);
  }

  /** Floor points a person placed (not accepted proposals) on the frame most of them share. */
  function evidencePoints(state) {
    const byFrame = new Map();
    for (const [name, point] of Object.entries(state.keypoints)) {
      if (!FLOOR_NAMES.includes(name) || point.source === "proposed") continue;
      const frame = point.timeMs ?? null;
      if (!byFrame.has(frame)) byFrame.set(frame, []);
      byFrame.get(frame).push(name);
    }
    let best = { frame: null, names: [] };
    byFrame.forEach((names, frame) => {
      if (names.length > best.names.length) best = { frame, names };
    });
    return best;
  }

  /** The court-to-image homography from the person's floor points on one frame, or null. */
  function courtHomography(state) {
    const { frame, names } = evidencePoints(state);
    const court = names.map(name => KEYPOINT_POSITIONS[name].slice(0, 2));
    if (names.length < 4 || !wellSpread(court)) return null;
    const h = homography(court, names.map(name => [state.keypoints[name].x, state.keypoints[name].y]));
    if (!h) return null;
    // The side of the camera the clicked points are on; points on the other side are behind it.
    return { h, sign: Math.sign(transformPoint(h, court[0])[2]), frame };
  }

  /**
   * A court looks the same mirrored or turned around, so wrongly named corners still fit a
   * perfect-looking court with every name wrong. Seen from above (any phone above the floor),
   * the singles corners back0_sl -> back0_sr -> back1_sr -> back1_sl run clockwise in the image,
   * and the back0 baseline is the one nearest the camera, so it looks longer.
   */
  function courtProblem(state) {
    const fit = courtHomography(state);
    if (!fit) return null;
    const corners = ["back0_sl", "back0_sr", "back1_sr", "back1_sl"].map(name => transformPoint(fit.h, KEYPOINT_POSITIONS[name]));
    let twiceArea = 0;
    for (let i = 0; i < 4; i += 1) {
      const [a, b] = [corners[i], corners[(i + 1) % 4]];
      twiceArea += a[0] * b[1] - b[0] * a[1];
    }
    const nearLength = Math.hypot(corners[0][0] - corners[1][0], corners[0][1] - corners[1][1]);
    const farLength = Math.hypot(corners[3][0] - corners[2][0], corners[3][1] - corners[2][1]);
    if (nearLength < 0.85 * farLength) {
      return "These points have near and far swapped: back0 is the baseline nearest the camera (it looks longer), back1 the far one.";
    }
    if (twiceArea >= 0) {
      return "These points have left and right swapped: sl is the left singles sideline as seen from the camera, sr the right.";
    }
    return null;
  }

  /** Image position of a court point through the fitted homography; null if behind the camera. */
  function projectCourtPoint(fit, position) {
    const [x, y, w] = transformPoint(fit.h, position);
    return Math.sign(w) === fit.sign && Number.isFinite(x) && Number.isFinite(y) ? [x, y] : null;
  }

  /** Proposed pixel positions for every unplaced floor point that lands inside the frame. */
  function proposeCourtPoints(state) {
    const fit = courtHomography(state);
    if (!fit || courtProblem(state)) return {};
    const [width, height] = state.imageSize;
    const proposals = {};
    for (const name of FLOOR_NAMES) {
      if (state.keypoints[name]) continue;
      const pixel = projectCourtPoint(fit, KEYPOINT_POSITIONS[name]);
      if (pixel && pixel[0] >= 0 && pixel[1] >= 0 && pixel[0] < width && pixel[1] < height) {
        proposals[name] = { x: round2(pixel[0]), y: round2(pixel[1]) };
      }
    }
    return proposals;
  }

  /** Keep proposals as keypoints marked "proposed", on the frame the person's points share. */
  function acceptCourtProposals(state, proposals) {
    const { frame } = evidencePoints(state);
    const names = Object.keys(proposals);
    for (const name of names) setKeypoint(state, name, proposals[name].x, proposals[name].y, frame ?? undefined, "proposed");
    return names.length;
  }

  /** After a person moves one of their points, move the untouched proposals with it. */
  function rederiveProposed(state) {
    const fit = courtHomography(state);
    if (!fit || courtProblem(state)) return 0;
    let moved = 0;
    for (const [name, point] of Object.entries(state.keypoints)) {
      if (point.source !== "proposed") continue;
      const pixel = projectCourtPoint(fit, KEYPOINT_POSITIONS[name]);
      if (!pixel) continue;
      const [x, y] = [round2(pixel[0]), round2(pixel[1])];
      if (x !== point.x || y !== point.y) moved += 1;
      setKeypoint(state, name, x, y, point.timeMs, "proposed");
    }
    return moved;
  }

  function invert3(m) {
    const [[a, b, c], [d, e, f], [g, h, i]] = m;
    const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    return [
      [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
      [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
      [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
    ];
  }

  /** Floor position (court metres) of an image pixel, through the fitted homography. */
  function imageToCourt(fit, pixel) {
    const [x, y] = transformPoint(invert3(fit.h), pixel);
    return [x, y];
  }

  // ---- Hits from audio ----

  const ONSET_FRAME_S = 0.0025;
  const ONSET_RISE = 0.5; // log10 of the energy ratio over the local noise floor: ~3x louder
  const ONSET_MIN_GAP_MS = 120;
  const ONSET_ATTACK_MS = 20; // a hit must also jump this much above the 20 ms just before it
  const ONSET_BLOCK_S = 0.5; // the noise floor is the median of +/- 2 such blocks around each frame
  const ONSET_FLOOR_MIN = -8; // log10 energy: digital silence must not make faint noise look like hits

  function median(values) {
    const sorted = Float64Array.from(values).sort();
    const middle = sorted.length >> 1;
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }

  // Second-order filter coefficients (RBJ audio-EQ cookbook, Q = 1/sqrt(2)).
  function biquad(type, frequency, sampleRate) {
    const w0 = (2 * Math.PI * frequency) / sampleRate;
    const cos = Math.cos(w0);
    const alpha = Math.sin(w0) / (2 * Math.SQRT1_2);
    const b = type === "highpass" ? [(1 + cos) / 2, -(1 + cos), (1 + cos) / 2] : [(1 - cos) / 2, 1 - cos, (1 - cos) / 2];
    const a0 = 1 + alpha;
    return { b0: b[0] / a0, b1: b[1] / a0, b2: b[2] / a0, a1: (-2 * cos) / a0, a2: (1 - alpha) / a0, x1: 0, x2: 0, y1: 0, y2: 0 };
  }

  function runBiquad(f, x) {
    const y = f.b0 * x + f.b1 * f.x1 + f.b2 * f.x2 - f.a1 * f.y1 - f.a2 * f.y2;
    f.x2 = f.x1;
    f.x1 = x;
    f.y2 = f.y1;
    f.y1 = y;
    return y;
  }

  /**
   * Sharp, bright onsets (racket hits) in mono samples. Per 2.5 ms frame we take the energy in
   * the 2-9 kHz band where the racket "thwack" lives (4th-order high- and low-pass, filtered as
   * the samples stream past), so voices, footsteps and broadband hiss count for little. An
   * onset is a frame ~3x louder than both the running noise floor around it and the 20 ms just
   * before it: the sharp attack is what tells a racket hit from a shout or a reverb tail, so
   * sustained sounds trigger at most once while a hit inside them is still found. Strength is
   * how much louder (log10) the peak is, so quieter hits from a neighbouring court rank lower.
   */
  function detectOnsets(samples, sampleRate, options = {}) {
    const frame = Math.max(2, Math.round(sampleRate * ONSET_FRAME_S));
    const count = Math.floor(samples.length / frame);
    if (count < 8) return [];
    const top = Math.min(9000, 0.45 * sampleRate);
    const filters = [biquad("highpass", 2000, sampleRate), biquad("highpass", 2000, sampleRate), biquad("lowpass", top, sampleRate), biquad("lowpass", top, sampleRate)];
    const level = new Float64Array(count);
    for (let f = 0; f < count; f += 1) {
      let sum = 0;
      for (let i = f * frame; i < (f + 1) * frame; i += 1) {
        let y = samples[i];
        for (const filter of filters) y = runBiquad(filter, y);
        sum += y * y;
      }
      level[f] = Math.log10(sum / frame + 1e-12);
    }
    // A running noise floor follows crowd noise and quiet passages instead of one clip-wide value.
    const block = Math.max(1, Math.round((ONSET_BLOCK_S * sampleRate) / frame));
    const blocks = Math.ceil(count / block);
    const blockFloor = Array.from({ length: blocks }, (_, b) => median(level.subarray(b * block, Math.min(count, (b + 1) * block))));
    const floorAt = new Float64Array(blocks);
    for (let b = 0; b < blocks; b += 1) {
      floorAt[b] = Math.max(ONSET_FLOOR_MIN, median(blockFloor.slice(Math.max(0, b - 2), Math.min(blocks, b + 3))));
    }
    const floor = f => floorAt[Math.floor(f / block)];
    const rise = options.rise ?? ONSET_RISE;
    const gap = Math.max(1, Math.round((((options.minGapMs ?? ONSET_MIN_GAP_MS) / 1000) * sampleRate) / frame));
    const attack = Math.max(1, Math.round(((ONSET_ATTACK_MS / 1000) * sampleRate) / frame));
    const peakWindow = Math.max(1, Math.round((0.03 * sampleRate) / frame));
    const onsets = [];
    let f = 0;
    while (f < count) {
      let before = 0;
      for (let g = Math.max(0, f - attack); g < f; g += 1) before += level[g];
      const recent = f > 0 ? before / (f - Math.max(0, f - attack)) : floor(f);
      if (level[f] - floor(f) < rise || level[f] - recent < rise) {
        f += 1;
        continue;
      }
      // The hit can start late in the previous frame if that frame is already clearly louder.
      const start = f > 0 && level[f - 1] - floor(f - 1) > rise * 0.3 ? f - 1 : f;
      let peak = level[f];
      for (let g = f; g < Math.min(count, f + peakWindow); g += 1) peak = Math.max(peak, level[g]);
      onsets.push({ timeMs: Math.round(((start * frame) / sampleRate) * 10000) / 10, strength: round2(peak - floor(f)) });
      f += gap;
    }
    return onsets;
  }

  /** Audio onsets become pending suggestions, except ones a labelled hit already explains. */
  function setHitSuggestions(state, onsets) {
    state.hitSuggestions = onsets
      .map(onset => ({ timeMs: Math.round(onset.timeMs), strength: onset.strength, status: "pending" }))
      .filter(s => !state.contacts.some(c => s.timeMs >= c.timeMs - MERGE_WINDOW_MS && s.timeMs <= c.timeMs + 80));
    return state.hitSuggestions.length;
  }

  /** Where to put the playhead for a suggestion: the sound arrives after the hit. */
  function suggestionSeekMs(suggestion) {
    return Math.max(0, suggestion.timeMs - SOUND_LEAD_MS);
  }

  function nextPendingSuggestion(state, afterIndex) {
    return state.hitSuggestions.findIndex((s, i) => i > afterIndex && s.status === "pending");
  }

  function acceptSuggestion(state, index, contactTimeMs, label) {
    const suggestion = state.hitSuggestions[index];
    if (!suggestion) throw new Error(`No suggestion at index ${index}`);
    if (Math.abs(contactTimeMs - suggestion.timeMs) > SUGGESTION_WINDOW_MS) {
      throw new Error(`That frame is far from the suggested hit at ${(suggestion.timeMs / 1000).toFixed(2)} s; press N to go back to it.`);
    }
    const contact = addContact(state, contactTimeMs, label);
    if (label) setLabel(state, contact, label);
    state.hitSuggestions[index].status = "accepted";
    return contact;
  }

  function dismissSuggestion(state, index) {
    if (!state.hitSuggestions[index]) throw new Error(`No suggestion at index ${index}`);
    state.hitSuggestions[index].status = "dismissed";
  }

  // ---- Rallies ----

  /** Hits more than RALLY_GAP_MS apart start a new rally (serve toss before, landing after). */
  function suggestRallies(contacts, gapMs = RALLY_GAP_MS) {
    const times = contacts.map(contact => contact.timeMs).sort((a, b) => a - b);
    const groups = [];
    for (const time of times) {
      const group = groups[groups.length - 1];
      if (group && time - group[group.length - 1] <= gapMs) group.push(time);
      else groups.push([time]);
    }
    return groups.map(group => ({ startMs: Math.max(0, group[0] - 600), endMs: group[group.length - 1] + 1500 }));
  }

  /** Suggested rallies stay unconfirmed (and out of the export) until a winner is chosen. */
  function applyRallySuggestions(state, durationMs) {
    let added = 0;
    for (const suggestion of suggestRallies(state.contacts)) {
      const endMs = durationMs ? Math.min(suggestion.endMs, Math.round(durationMs)) : suggestion.endMs;
      const overlaps = state.rallies.some(rally => suggestion.startMs < (rally.endMs ?? Infinity) && rally.startMs < endMs);
      if (overlaps || endMs <= suggestion.startMs) continue;
      state.rallies.push({ startMs: suggestion.startMs, endMs, winner: "unknown", confirmed: false });
      added += 1;
    }
    state.rallies.sort((a, b) => a.startMs - b.startMs);
    return added;
  }

  /** Choosing a winner (including "unknown") is what confirms a suggested rally. */
  function setRallyWinner(state, index, winner) {
    if (!WINNERS.includes(winner)) throw new Error(`Unknown winner: ${winner}`);
    if (!state.rallies[index]) throw new Error(`No rally at index ${index}`);
    state.rallies[index].winner = winner;
    delete state.rallies[index].confirmed;
  }

  function setRallyBounds(state, index, startMs, endMs) {
    const rally = state.rallies[index];
    if (!rally) throw new Error(`No rally at index ${index}`);
    const start = Math.round(finite(startMs, "Rally start"));
    const end = Math.round(finite(endMs, "Rally end"));
    if (end <= start) throw new Error("A rally must end after it starts.");
    rally.startMs = start;
    rally.endMs = end;
  }

  /** The rally containing a time, else the latest rally that started before it, else -1. */
  function rallyIndexAt(state, timeMs) {
    let latest = -1;
    for (let i = 0; i < state.rallies.length; i += 1) {
      const rally = state.rallies[i];
      if (rally.startMs <= timeMs && timeMs <= (rally.endMs ?? Infinity)) return i;
      if (rally.startMs <= timeMs && (latest < 0 || rally.startMs > state.rallies[latest].startMs)) latest = i;
    }
    return latest;
  }

  // ---- Body poses: the selected player's 17 COCO joints at each hit frame ----

  function setSelectedPlayer(state, player) {
    if (!PLAYERS.includes(player)) throw new Error(`Unknown player: ${player}`);
    if (Object.values(state.poses).some(pose => pose.player !== player)) {
      throw new Error("Delete the other player's poses before switching player: a clip labels one player.");
    }
    state.selectedPlayer = player;
  }

  /** A detected pose belongs to one frame; confirming it on any other frame would mislabel it. */
  function canConfirmPose(pose, frameMs) {
    return Boolean(pose && pose.keypoints && pose.timeMs === frameMs);
  }

  /** Right-click cycle: visible -> hidden (placed but not seen) -> not labelled -> visible. */
  function nextVisibility(visibility) {
    return visibility === 2 ? 1 : visibility === 1 ? 0 : 2;
  }

  function setPose(state, timeMs, player, keypoints) {
    if (!PLAYERS.includes(player)) throw new Error(`Unknown player: ${player}`);
    if (!Array.isArray(keypoints) || keypoints.length !== COCO_JOINTS.length) {
      throw new Error(`A pose needs all ${COCO_JOINTS.length} COCO joints`);
    }
    const joints = keypoints.map(([x, y, v], i) => {
      if (![0, 1, 2].includes(v)) throw new Error(`Joint ${COCO_JOINTS[i]} visibility must be 0, 1 or 2`);
      return [round2(finite(x, "Joint x")), round2(finite(y, "Joint y")), v];
    });
    state.poses[String(Math.round(finite(timeMs, "Pose time")))] = { player, keypoints: joints };
  }

  function removePose(state, timeMs) {
    delete state.poses[String(Math.round(timeMs))];
  }

  /** Hit times that still need the selected player's pose. */
  function poseTargets(state) {
    return state.contacts.map(contact => contact.timeMs).filter(time => !state.poses[String(time)]);
  }

  /**
   * Pose-model output ({keypoints: [{x, y, score}], score}) as candidates with [x, y, visibility]
   * joints. Low-confidence joints are model guesses, not labels: they start unlabelled (0) and
   * become labels only when a person drags them. Joints are kept inside the frame.
   */
  function candidatesFromDetections(detections, minJointScore = 0.3, imageSize = null) {
    const clamp = (value, size) => (size ? Math.min(Math.max(value, 0), size - 0.01) : value);
    return detections
      .filter(detection => Array.isArray(detection.keypoints) && detection.keypoints.length === COCO_JOINTS.length)
      .map(detection => {
        const keypoints = detection.keypoints.map(joint => [
          round2(clamp(joint.x, imageSize && imageSize[0])),
          round2(clamp(joint.y, imageSize && imageSize[1])),
          (joint.score ?? 0) >= minJointScore ? 2 : 0,
        ]);
        return { keypoints, score: detection.score ?? 0, bottom: Math.max(...keypoints.map(joint => joint[1])) };
      });
  }

  function centre(keypoints) {
    const sum = keypoints.reduce((acc, joint) => [acc[0] + joint[0], acc[1] + joint[1]], [0, 0]);
    return [sum[0] / keypoints.length, sum[1] / keypoints.length];
  }

  /** Where a candidate stands: the midpoint of their ankles, else their lowest joint. */
  function feetOf(candidate) {
    const ankles = [candidate.keypoints[15], candidate.keypoints[16]].filter(joint => joint[2] > 0);
    if (ankles.length) return [ankles.reduce((s, j) => s + j[0], 0) / ankles.length, ankles.reduce((s, j) => s + j[1], 0) / ankles.length];
    const lowest = candidate.keypoints.reduce((a, b) => (b[1] > a[1] ? b : a));
    return [lowest[0], lowest[1]];
  }

  /**
   * The candidate that is the selected player. With a court fit, only people standing on that
   * player's half are considered (when anyone is). Among those: the one nearest their last
   * confirmed pose, or the lowest in the frame for the near player and the highest for the far one.
   */
  function pickPlayer(candidates, player, previousKeypoints, fit) {
    if (!candidates.length) return -1;
    if (fit) {
      const onSide = candidates
        .map((candidate, index) => ({ candidate, index }))
        .filter(({ candidate }) => {
          const [, y] = imageToCourt(fit, feetOf(candidate));
          return player === "near" ? y < NET_Y_M : y > NET_Y_M;
        });
      if (onSide.length && onSide.length < candidates.length) {
        const pick = pickPlayer(onSide.map(item => item.candidate), player, previousKeypoints, null);
        return onSide[pick].index;
      }
    }
    let best = 0;
    if (previousKeypoints) {
      const target = centre(previousKeypoints);
      const distance = candidate => {
        const c = centre(candidate.keypoints);
        return Math.hypot(c[0] - target[0], c[1] - target[1]);
      };
      candidates.forEach((candidate, i) => {
        if (distance(candidate) < distance(candidates[best])) best = i;
      });
      return best;
    }
    candidates.forEach((candidate, i) => {
      const better = player === "near" ? candidate.bottom > candidates[best].bottom : candidate.bottom < candidates[best].bottom;
      if (better) best = i;
    });
    return best;
  }

  /** The selected player's most recent confirmed pose before a time, or null. */
  function previousPose(state, timeMs) {
    let found = null;
    let foundTime = -Infinity;
    for (const [time, pose] of Object.entries(state.poses)) {
      const t = Number(time);
      if (t < timeMs && t > foundTime && pose.player === state.selectedPlayer) {
        found = pose.keypoints;
        foundTime = t;
      }
    }
    return found;
  }

  function importWorkerResult(state, result) {
    const shots = result.shots || [];
    let added = 0;
    for (const event of result.events || []) {
      if (event.type !== "contact" || typeof event.timeMs !== "number") continue;
      const shot = shots.find(candidate => Math.abs(candidate.timeMs - event.timeMs) <= 100);
      const label = shot && shot.verified && SHOT_LABELS.includes(shot.label) ? shot.label : "other";
      const before = state.contacts.length;
      addContact(state, event.timeMs, label);
      if (state.contacts.length > before) added += 1;
    }
    return added;
  }

  function buildGolden(state) {
    const warnings = [];
    const rallies = [];
    let unchecked = 0;
    for (const rally of state.rallies) {
      if (rally.endMs === null) {
        warnings.push(`Rally starting at ${rally.startMs} ms has no end and was left out of the export.`);
        continue;
      }
      if (rally.confirmed === false) {
        unchecked += 1;
        continue;
      }
      rallies.push({ startMs: rally.startMs, endMs: rally.endMs, winner: rally.winner });
    }
    if (unchecked > 0) warnings.push(`${unchecked} suggested rally(s) not checked yet: choose a winner to confirm each one.`);
    const unlabelled = state.contacts.filter(contact => contact.label === "other").length;
    if (unlabelled > 0) warnings.push(`${unlabelled} contact(s) are still labelled "other".`);
    const missingPoses = state.selectedPlayer ? poseTargets(state).length : 0;
    if (missingPoses > 0) warnings.push(`${missingPoses} hit frame(s) still need a body pose.`);
    const doc = {
      schemaVersion: 1,
      clipId: state.clipId,
      sourceFps: state.fps,
      imageSize: [state.imageSize[0], state.imageSize[1]],
      courtKeypoints: Object.entries(state.keypoints).map(([name, point]) => {
        const item = { name, x: point.x, y: point.y };
        if (point.timeMs !== undefined) item.timeMs = point.timeMs;
        if (point.source !== undefined) item.source = point.source;
        return item;
      }),
      contacts: state.contacts.map(contact => ({ timeMs: contact.timeMs })),
      shots: state.contacts.map(contact => ({ timeMs: contact.timeMs, label: contact.label, landing: null })),
      rallies,
    };
    if (state.selectedPlayer) doc.selectedPlayer = state.selectedPlayer;
    const contactTimes = new Set(state.contacts.map(contact => contact.timeMs));
    const allPoses = Object.entries(state.poses).map(([time, pose]) => ({ timeMs: Number(time), player: pose.player, keypoints: pose.keypoints }));
    const orphans = allPoses.filter(pose => !contactTimes.has(pose.timeMs)).length;
    if (orphans > 0) warnings.push(`${orphans} pose(s) are not on a hit and were left out of the export.`);
    const poses = allPoses.filter(pose => contactTimes.has(pose.timeMs)).sort((a, b) => a.timeMs - b.timeMs);
    if (poses.length) doc.poses = poses;
    return { doc, warnings };
  }

  function stateFromGolden(doc) {
    const state = emptyState(doc.clipId, doc.sourceFps, doc.imageSize);
    for (const item of doc.courtKeypoints || []) {
      const point = { x: item.x, y: item.y };
      if (item.timeMs !== undefined) point.timeMs = item.timeMs;
      if (item.source !== undefined) point.source = item.source;
      state.keypoints[item.name] = point;
    }
    const labelAt = new Map((doc.shots || []).map(shot => [shot.timeMs, shot.label]));
    for (const contact of doc.contacts || []) {
      state.contacts.push({ timeMs: contact.timeMs, label: labelAt.get(contact.timeMs) || "other" });
    }
    state.contacts.sort((a, b) => a.timeMs - b.timeMs);
    for (const rally of doc.rallies || []) {
      state.rallies.push({ startMs: rally.startMs, endMs: rally.endMs, winner: rally.winner });
    }
    state.selectedPlayer = doc.selectedPlayer || null;
    for (const pose of doc.poses || []) state.poses[String(pose.timeMs)] = { player: pose.player, keypoints: pose.keypoints };
    return state;
  }

  // ---- Loading files safely ----

  const isNumber = value => typeof value === "number" && Number.isFinite(value);

  /** The checks of worker/evaluation/golden.py validate_golden, for files loaded in the browser. */
  function validateGolden(doc) {
    if (!doc || typeof doc !== "object" || Array.isArray(doc)) return ["document must be a JSON object"];
    const errors = [];
    if (doc.schemaVersion !== 1) errors.push("schemaVersion must be 1");
    if (typeof doc.clipId !== "string" || !doc.clipId.trim()) errors.push("clipId must be a non-empty string");
    if (!isNumber(doc.sourceFps) || doc.sourceFps <= 0) errors.push("sourceFps must be a positive number");
    const size = doc.imageSize;
    const sizeOk = Array.isArray(size) && size.length === 2 && size.every(side => Number.isInteger(side) && side > 0);
    if (!sizeOk) errors.push("imageSize must be [width, height] positive integers");
    const inside = (x, y) => !sizeOk || (x >= 0 && x < size[0] && y >= 0 && y < size[1]);
    const points = doc.courtKeypoints === undefined ? [] : doc.courtKeypoints;
    if (!Array.isArray(points)) errors.push("courtKeypoints must be a list");
    (Array.isArray(points) ? points : []).forEach((item, i) => {
      const where = `courtKeypoints[${i}]`;
      if (!item || typeof item !== "object" || !KEYPOINT_NAMES.includes(item.name)) errors.push(`${where}: keypoint name must be one of the court model names`);
      else if (!isNumber(item.x) || !isNumber(item.y)) errors.push(`${where}: x and y must be numbers`);
      else if (!inside(item.x, item.y)) errors.push(`${where}: pixel is outside the image`);
      if (item && typeof item === "object" && "timeMs" in item && (!isNumber(item.timeMs) || item.timeMs < 0)) errors.push(`${where}: timeMs must be a non-negative number`);
      if (item && typeof item === "object" && "source" in item && !KEYPOINT_SOURCES.includes(item.source)) errors.push(`${where}: source must be clicked, adjusted or proposed`);
    });
    for (const key of ["contacts", "shots"]) if (!Array.isArray(doc[key])) errors.push(`${key} must be a list`);
    const contacts = Array.isArray(doc.contacts) ? doc.contacts : [];
    contacts.forEach((item, i) => {
      if (!item || !isNumber(item.timeMs) || item.timeMs < 0) errors.push(`contacts[${i}]: timeMs must be a non-negative number`);
    });
    (Array.isArray(doc.shots) ? doc.shots : []).forEach((item, i) => {
      if (!item || !isNumber(item.timeMs) || item.timeMs < 0) { errors.push(`shots[${i}]: timeMs must be a non-negative number`); return; }
      if (!SHOT_LABELS.includes(item.label)) errors.push(`shots[${i}]: unknown label`);
      const landing = item.landing;
      if (landing !== null && landing !== undefined) {
        const ok = typeof landing === "object" && isNumber(landing.x) && isNumber(landing.y) &&
          landing.x >= -DOUBLES_MARGIN_M - 1 && landing.x <= SINGLES_WIDTH_M + DOUBLES_MARGIN_M + 1 && landing.y >= -1 && landing.y <= COURT_LENGTH_M + 1;
        if (!ok) errors.push(`shots[${i}]: landing must be null or court metres near the court`);
      }
    });
    const rallies = doc.rallies === undefined ? [] : doc.rallies;
    (Array.isArray(rallies) ? rallies : []).forEach((item, i) => {
      if (!item || !isNumber(item.startMs) || !isNumber(item.endMs)) { errors.push(`rallies[${i}]: startMs and endMs must be numbers`); return; }
      if (item.startMs >= item.endMs) errors.push(`rallies[${i}]: startMs must be before endMs`);
      if (!WINNERS.includes(item.winner)) errors.push(`rallies[${i}]: unknown winner`);
    });
    if ("selectedPlayer" in doc && !PLAYERS.includes(doc.selectedPlayer)) errors.push("selectedPlayer must be near or far when present");
    const contactTimes = new Set(contacts.filter(c => c && isNumber(c.timeMs)).map(c => c.timeMs));
    const seen = new Set();
    (Array.isArray(doc.poses) ? doc.poses : []).forEach((item, i) => {
      const where = `poses[${i}]`;
      if (!item || typeof item !== "object" || !isNumber(item.timeMs) || item.timeMs < 0) { errors.push(`${where}: timeMs must be a non-negative number`); return; }
      if (!PLAYERS.includes(item.player)) errors.push(`${where}: player must be near or far`);
      if (!Array.isArray(item.keypoints) || item.keypoints.length !== COCO_JOINTS.length) errors.push(`${where}: keypoints must list the 17 COCO joints`);
      else item.keypoints.forEach((joint, j) => {
        const ok = Array.isArray(joint) && joint.length === 3 && joint.every(isNumber) && [0, 1, 2].includes(joint[2]);
        if (!ok) errors.push(`${where}.${COCO_JOINTS[j]}: must be [x, y, visibility 0|1|2]`);
        else if (joint[2] > 0 && !inside(joint[0], joint[1])) errors.push(`${where}.${COCO_JOINTS[j]}: labelled joint is outside the image`);
      });
      if (!contactTimes.has(item.timeMs)) errors.push(`${where}: timeMs must be the time of a labelled contact`);
      if (seen.has(item.timeMs)) errors.push(`${where}: duplicate pose`);
      seen.add(item.timeMs);
      if (PLAYERS.includes(doc.selectedPlayer) && PLAYERS.includes(item.player) && item.player !== doc.selectedPlayer) errors.push(`${where}: player must be the selectedPlayer`);
    });
    return errors;
  }

  function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);
  }

  /** An autosave belongs to one video file: same name, size, date, dimensions and length. */
  function autosaveMatches(saved, fingerprint) {
    const stored = saved && saved.fingerprint;
    if (!stored || !fingerprint) return false;
    return ["name", "size", "lastModified", "width", "height"].every(key => stored[key] === fingerprint[key]) &&
      Math.abs((stored.durationS ?? -1) - fingerprint.durationS) < 0.05;
  }

  // Video time <-> frames. A time anywhere within ~a quarter frame of a frame start or its
  // centre maps to that frame, which tolerates timestamp jitter in phone footage.
  function frameIndexAt(seconds, fps) {
    return Math.max(0, Math.floor(seconds * fps + 0.25));
  }

  function frameTimeMs(index, fps) {
    return Math.round((index * 1000) / fps);
  }

  function frameCentreSeconds(index, fps) {
    return (index + 0.5) / fps;
  }

  function stepSeconds(currentSeconds, frames, fps, durationSeconds) {
    const last = Math.max(0, Math.ceil(durationSeconds * fps) - 1);
    const index = Math.min(last, Math.max(0, frameIndexAt(currentSeconds, fps) + frames));
    return frameCentreSeconds(index, fps);
  }

  function fpsFromMediaTimes(times) {
    const deltas = [];
    for (let i = 1; i < times.length; i += 1) {
      const delta = times[i] - times[i - 1];
      if (delta > 0) deltas.push(delta);
    }
    if (deltas.length < 2) return null;
    deltas.sort((a, b) => a - b);
    return Math.round(1000 / deltas[Math.floor(deltas.length / 2)]) / 1000;
  }

  return {
    SHOT_LABELS,
    KEYPOINT_NAMES,
    KEYPOINT_POSITIONS,
    FLOOR_NAMES,
    COURT_LINES,
    COCO_JOINTS,
    SKELETON,
    PLAYERS,
    SOUND_LEAD_MS,
    emptyState,
    normalizeState,
    addContact,
    setLabel,
    removeContact,
    nearestContactIndex,
    startRally,
    endRally,
    setWinner,
    setKeypoint,
    removeKeypoint,
    courtHomography,
    courtProblem,
    projectCourtPoint,
    imageToCourt,
    proposeCourtPoints,
    acceptCourtProposals,
    moveKeypoint,
    rederiveProposed,
    detectOnsets,
    setHitSuggestions,
    suggestionSeekMs,
    nextPendingSuggestion,
    acceptSuggestion,
    dismissSuggestion,
    suggestRallies,
    applyRallySuggestions,
    setRallyWinner,
    setRallyBounds,
    rallyIndexAt,
    setSelectedPlayer,
    canConfirmPose,
    nextVisibility,
    setPose,
    removePose,
    poseTargets,
    candidatesFromDetections,
    pickPlayer,
    previousPose,
    importWorkerResult,
    buildGolden,
    stateFromGolden,
    validateGolden,
    escapeHtml,
    autosaveMatches,
    frameIndexAt,
    frameTimeMs,
    frameCentreSeconds,
    stepSeconds,
    fpsFromMediaTimes,
  };
});
