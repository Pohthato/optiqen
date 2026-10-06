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
    state.contacts.splice(index, 1);
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
  function setKeypoint(state, name, x, y, timeMs) {
    if (!KEYPOINT_NAMES.includes(name)) throw new Error(`Unknown court keypoint: ${name}`);
    const point = { x: round2(finite(x, "Pixel x")), y: round2(finite(y, "Pixel y")) };
    if (timeMs !== undefined) point.timeMs = Math.round(finite(timeMs, "Keypoint time"));
    state.keypoints[name] = point;
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

  // Points that do not all lie near one line: some triangle covers more than 0.5 square metres.
  function wellSpread(points) {
    for (let i = 0; i < points.length; i += 1)
      for (let j = i + 1; j < points.length; j += 1)
        for (let k = j + 1; k < points.length; k += 1) {
          const [a, b, c] = [points[i], points[j], points[k]];
          if (Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2 > 0.5) return true;
        }
    return false;
  }

  /** The court-to-image homography from the clicked floor points, or null. */
  function courtHomography(state) {
    const names = Object.keys(state.keypoints).filter(name => FLOOR_NAMES.includes(name));
    const court = names.map(name => KEYPOINT_POSITIONS[name].slice(0, 2));
    if (names.length < 4 || !wellSpread(court)) return null;
    const h = homography(court, names.map(name => [state.keypoints[name].x, state.keypoints[name].y]));
    if (!h) return null;
    // The side of the camera the clicked points are on; points on the other side are behind it.
    return { h, sign: Math.sign(transformPoint(h, court[0])[2]) };
  }

  /** Image position of a court point through the fitted homography; null if behind the camera. */
  function projectCourtPoint(fit, position) {
    const [x, y, w] = transformPoint(fit.h, position);
    return Math.sign(w) === fit.sign && Number.isFinite(x) && Number.isFinite(y) ? [x, y] : null;
  }

  /** Proposed pixel positions for every unclicked floor point that lands inside the frame. */
  function proposeCourtPoints(state) {
    const fit = courtHomography(state);
    if (!fit) return {};
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

  /** Keep proposals as keypoints, stamped with the frame most clicked points came from. */
  function acceptCourtProposals(state, proposals) {
    const counts = new Map();
    for (const point of Object.values(state.keypoints)) {
      if (point.timeMs !== undefined) counts.set(point.timeMs, (counts.get(point.timeMs) || 0) + 1);
    }
    let timeMs;
    counts.forEach((count, time) => {
      if (timeMs === undefined || count > counts.get(timeMs)) timeMs = time;
    });
    const names = Object.keys(proposals);
    for (const name of names) setKeypoint(state, name, proposals[name].x, proposals[name].y, timeMs);
    return names.length;
  }

  // ---- Hits from audio ----

  const ONSET_FRAME_S = 0.0025;
  const ONSET_RISE = 1.0; // log10 of the energy ratio over the noise floor: 10x louder
  const ONSET_MIN_GAP_MS = 120;

  function median(values) {
    const sorted = Float64Array.from(values).sort();
    const middle = sorted.length >> 1;
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }

  /**
   * Sharp, bright onsets (racket hits) in mono samples. Per 2.5 ms frame we take the energy
   * of the sample-to-sample difference, which favours the high-frequency "thwack" over voices
   * and footsteps. An onset is a frame at least 10x louder than the clip's median frame;
   * its strength is how much louder (log10) its peak is, so quieter hits from a neighbouring
   * court rank below the rally's own.
   */
  function detectOnsets(samples, sampleRate, options = {}) {
    const frame = Math.max(2, Math.round(sampleRate * ONSET_FRAME_S));
    const count = Math.floor(samples.length / frame);
    if (count < 8) return [];
    const level = new Float64Array(count);
    for (let f = 0; f < count; f += 1) {
      let sum = 0;
      for (let i = f * frame + 1; i < (f + 1) * frame; i += 1) {
        const d = samples[i] - samples[i - 1];
        sum += d * d;
      }
      level[f] = Math.log10(sum / (frame - 1) + 1e-12);
    }
    const floor = median(level);
    const rise = options.rise ?? ONSET_RISE;
    const gap = Math.max(1, Math.round((((options.minGapMs ?? ONSET_MIN_GAP_MS) / 1000) * sampleRate) / frame));
    const peakWindow = Math.max(1, Math.round((0.03 * sampleRate) / frame));
    const onsets = [];
    let f = 0;
    while (f < count) {
      if (level[f] - floor < rise) {
        f += 1;
        continue;
      }
      // The hit can start late in the previous frame if that frame is already clearly louder.
      const start = f > 0 && level[f - 1] - floor > rise * 0.3 ? f - 1 : f;
      let peak = level[f];
      for (let g = f; g < Math.min(count, f + peakWindow); g += 1) peak = Math.max(peak, level[g]);
      onsets.push({ timeMs: Math.round(((start * frame) / sampleRate) * 10000) / 10, strength: round2(peak - floor) });
      f += gap;
      while (f < count && level[f] - floor >= rise * 0.5) f += 1;
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
    if (!state.hitSuggestions[index]) throw new Error(`No suggestion at index ${index}`);
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

  function applyRallySuggestions(state) {
    let added = 0;
    for (const suggestion of suggestRallies(state.contacts)) {
      const overlaps = state.rallies.some(rally => suggestion.startMs < (rally.endMs ?? Infinity) && rally.startMs < suggestion.endMs);
      if (overlaps) continue;
      state.rallies.push({ ...suggestion, winner: "unknown" });
      added += 1;
    }
    state.rallies.sort((a, b) => a.startMs - b.startMs);
    return added;
  }

  function setRallyWinner(state, index, winner) {
    if (!WINNERS.includes(winner)) throw new Error(`Unknown winner: ${winner}`);
    if (!state.rallies[index]) throw new Error(`No rally at index ${index}`);
    state.rallies[index].winner = winner;
  }

  // ---- Body poses: the selected player's 17 COCO joints at each hit frame ----

  function setSelectedPlayer(state, player) {
    if (!PLAYERS.includes(player)) throw new Error(`Unknown player: ${player}`);
    state.selectedPlayer = player;
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

  /** Pose-model output ({keypoints: [{x, y, score}], score}) as candidates with [x, y, visibility] joints. */
  function candidatesFromDetections(detections, minJointScore = 0.3) {
    return detections
      .filter(detection => Array.isArray(detection.keypoints) && detection.keypoints.length === COCO_JOINTS.length)
      .map(detection => {
        const keypoints = detection.keypoints.map(joint => [round2(joint.x), round2(joint.y), (joint.score ?? 0) >= minJointScore ? 2 : 1]);
        return { keypoints, score: detection.score ?? 0, bottom: Math.max(...keypoints.map(joint => joint[1])) };
      });
  }

  function centre(keypoints) {
    const sum = keypoints.reduce((acc, joint) => [acc[0] + joint[0], acc[1] + joint[1]], [0, 0]);
    return [sum[0] / keypoints.length, sum[1] / keypoints.length];
  }

  /**
   * The candidate that is the selected player: the one nearest their last confirmed pose, or
   * without one, the lowest in the frame for the near player and the highest for the far one.
   */
  function pickPlayer(candidates, player, previousKeypoints) {
    if (!candidates.length) return -1;
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
    for (const rally of state.rallies) {
      if (rally.endMs === null) {
        warnings.push(`Rally starting at ${rally.startMs} ms has no end and was left out of the export.`);
        continue;
      }
      rallies.push({ startMs: rally.startMs, endMs: rally.endMs, winner: rally.winner });
    }
    const unlabelled = state.contacts.filter(contact => contact.label === "other").length;
    if (unlabelled > 0) warnings.push(`${unlabelled} contact(s) are still labelled "other".`);
    const missingPoses = state.selectedPlayer ? poseTargets(state).length : 0;
    if (missingPoses > 0) warnings.push(`${missingPoses} hit frame(s) still need a body pose.`);
    const doc = {
      schemaVersion: 1,
      clipId: state.clipId,
      sourceFps: state.fps,
      imageSize: [state.imageSize[0], state.imageSize[1]],
      courtKeypoints: Object.entries(state.keypoints).map(([name, point]) =>
        point.timeMs === undefined ? { name, x: point.x, y: point.y } : { name, x: point.x, y: point.y, timeMs: point.timeMs }),
      contacts: state.contacts.map(contact => ({ timeMs: contact.timeMs })),
      shots: state.contacts.map(contact => ({ timeMs: contact.timeMs, label: contact.label, landing: null })),
      rallies,
    };
    if (state.selectedPlayer) doc.selectedPlayer = state.selectedPlayer;
    const poses = Object.entries(state.poses)
      .map(([time, pose]) => ({ timeMs: Number(time), player: pose.player, keypoints: pose.keypoints }))
      .sort((a, b) => a.timeMs - b.timeMs);
    if (poses.length) doc.poses = poses;
    return { doc, warnings };
  }

  function stateFromGolden(doc) {
    const state = emptyState(doc.clipId, doc.sourceFps, doc.imageSize);
    for (const item of doc.courtKeypoints || []) {
      state.keypoints[item.name] = item.timeMs === undefined ? { x: item.x, y: item.y } : { x: item.x, y: item.y, timeMs: item.timeMs };
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
    projectCourtPoint,
    proposeCourtPoints,
    acceptCourtProposals,
    detectOnsets,
    setHitSuggestions,
    suggestionSeekMs,
    nextPendingSuggestion,
    acceptSuggestion,
    dismissSuggestion,
    suggestRallies,
    applyRallySuggestions,
    setRallyWinner,
    setSelectedPlayer,
    setPose,
    removePose,
    poseTargets,
    candidatesFromDetections,
    pickPlayer,
    previousPose,
    importWorkerResult,
    buildGolden,
    stateFromGolden,
    frameIndexAt,
    frameTimeMs,
    frameCentreSeconds,
    stepSeconds,
    fpsFromMediaTimes,
  };
});
