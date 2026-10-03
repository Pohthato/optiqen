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
  const MERGE_WINDOW_MS = 20;

  const round2 = n => Math.round(n * 100) / 100;

  // A collapsed window or hidden tab gives the page zero-size layout, so coordinate maths
  // can yield NaN/Infinity. JSON.stringify would turn those into null in the export.
  function finite(value, what) {
    if (!Number.isFinite(value)) throw new Error(`${what} must be a finite number`);
    return value;
  }

  function emptyState(clipId, fps, imageSize) {
    return { clipId, fps, imageSize: [imageSize[0], imageSize[1]], keypoints: {}, contacts: [], rallies: [] };
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
    emptyState,
    addContact,
    setLabel,
    removeContact,
    nearestContactIndex,
    startRally,
    endRally,
    setWinner,
    setKeypoint,
    removeKeypoint,
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
