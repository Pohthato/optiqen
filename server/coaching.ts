// server/coaching.ts
// Coaching from the rally analysis's findings (worker/findings.py). The coach is given the findings
// and nothing else to make claims from, and must name the findings behind every sentence. A
// sentence is not shown when it names none, names one that does not exist, gives a number none of
// its findings holds, states as fact a finding left too close to call, or strays into injury or
// medicine (spec section 2).
import { z } from "zod";

export const TOO_CLOSE = "too close to call";

export const findingSchema = z.object({
  id: z.string().regex(/^F\d+$/),
  kind: z.enum(["end", "how", "turn", "valueTurn", "shot", "player"]),
  about: z.enum(["near", "far"]).nullable(),
  shots: z.array(z.number().int().positive()).max(400),
  evidence: z.array(z.object({ frame: z.number().int().nonnegative(), timeMs: z.number() })).min(1).max(400),
  call: z.string().min(1).max(64),
  p: z.number().min(0).max(1).nullable(),
  facts: z.record(z.string(), z.unknown()),
  text: z.string().min(1).max(600),
});

export type Finding = z.infer<typeof findingSchema>;

// The rally analysis as the worker returns it: its findings, and whatever else it holds.
export const rallySchema = z.looseObject({ findings: z.array(findingSchema).max(400) });

export type Sentence = { text: string; cites: string[]; why: string | null };

export type CheckedCoaching = { shown: Sentence[]; dropped: Sentence[] };

const CITE = /\[\s*(F\d+(?:\s*[,;]\s*F\d+)*)\s*\]/g;
const LEADING_CITE = /^(\[\s*F\d+(?:\s*[,;]\s*F\d+)*\s*\][.!?]?)\s*(.*)$/;
const NUMBER = /\d+(?:\.\d+)?/g;
const MEDICAL = /\b(injur\w*|pain\w*|strain\w*|sprain\w*|tendon\w*|tendin\w*|physio\w*|doctor\w*|diagnos\w*|medical\w*|medicin\w*|rehab\w*|sore\w*|ache\w*|inflam\w*|surgery)\b/i;
const UNSURE = /too close|uncertain|unclear|not sure|not certain|can ?not (?:tell|say|be sure)|can't (?:tell|say|be sure)|\bmay\b|\bmight\b|possibly|perhaps|likely|lean|chance|\d\s*%/i;

/** The findings a result carries, or null when it has none (an analysis from before them). */
export function findingsOf(result: unknown): Finding[] | null {
  const rally = (result as { rally?: unknown } | null | undefined)?.rally;
  const parsed = rallySchema.safeParse(rally);
  return parsed.success && parsed.data.findings.length ? parsed.data.findings : null;
}

export function buildFindingsPrompt(player: "near" | "far", findings: Finding[], question?: string): string {
  const given = findings.map(({ id, kind, about, shots, call, p, text, facts }) => ({ id, kind, about, shots, call, p, text, facts }));
  return [
    "You are Optiqen Coach. You coach one badminton player from what a video analysis found in their rally, and from nothing else.",
    "The findings below are all you know. Each has an ID, the player it is about, the shots it concerns, its call with its chance (p) or 'too close to call', its facts and a plain sentence that says it. A measured number comes with its standard deviation (sigma); a fact whose sigma is as large as the number tells little.",
    "Rules for every sentence you write:",
    "1. End it with the IDs of the findings it rests on, in square brackets, as [F3] or [F3, F7]. A sentence with no finding behind it is not shown.",
    "2. Use a number only if a finding you cite holds it, as it is there or rounded. Refer to shots by their number (shot 4). Write counts and repetitions in words.",
    "3. When a finding is too close to call, say that it is uncertain. Never state it as fact.",
    "4. Do not diagnose injuries, mention pain or give medical advice.",
    "5. Advice, such as what to practise, must also cite the findings that call for it.",
    "Write plain sentences in short paragraphs, with no headings, lists or Markdown, and at most twelve sentences.",
    `The player you coach is the ${player} player.`,
    question ? `Their question: ${question}` : "Give them a first assessment of this rally: what decided it, what they did well, and the one or two things that would help them most.",
    `Findings: ${JSON.stringify(given)}`,
  ].join("\n");
}

/** A reply cut into sentences, each with the cite that closes it ("… net. [F3]" keeps its [F3]). */
export function sentencesOf(reply: string): string[] {
  const out: string[] = [];
  for (const line of reply.split(/\r?\n+/)) {
    for (const raw of line.split(/(?<=[.!?])\s+(?=[[("'A-Z0-9])/)) {
      const part = raw.trim();
      if (!part) continue;
      const lead = part.match(LEADING_CITE);
      if (lead && out.length) {
        out[out.length - 1] += ` ${lead[1]}`;
        if (lead[2]) out.push(lead[2]);
      } else {
        out.push(part);
      }
    }
  }
  return out;
}

/** Every number a finding holds, as it may be said: as it is, a chance as a percentage, a time in
 * milliseconds as seconds, and the numbers its own sentence uses. */
function numbersOf(finding: Finding): number[] {
  const out: number[] = [...finding.shots];
  for (const { frame, timeMs } of finding.evidence) out.push(frame, timeMs, timeMs / 1000);
  const walk = (value: unknown, key: string) => {
    if (typeof value === "number") {
      out.push(Math.abs(value));
      if (value >= 0 && value <= 1) out.push(value * 100);
      if (/Ms$/.test(key)) out.push(Math.abs(value) / 1000);
    } else if (Array.isArray(value)) {
      value.forEach((item) => walk(item, key));
    } else if (value && typeof value === "object") {
      // A measurement's value and sigma are said in the unit of the key that holds them.
      Object.entries(value).forEach(([inner, item]) => walk(item, inner === "value" || inner === "sigma" || inner === "interval" ? key : inner));
    }
  };
  walk(finding.p, "p");
  walk(finding.facts, "facts");
  for (const match of Array.from(finding.text.matchAll(NUMBER))) out.push(Number(match[0]));
  return out;
}

function holds(numbers: number[], token: string): boolean {
  const decimals = token.includes(".") ? token.length - token.indexOf(".") - 1 : 0;
  const said = Number(token);
  const tolerance = 0.5 * 10 ** -decimals + 1e-9;
  return numbers.some((number) => Math.abs(number - said) <= tolerance);
}

/** Each sentence of a reply, shown or dropped, and why. */
export function checkCoaching(reply: string, findings: Finding[]): CheckedCoaching {
  const byId = new Map(findings.map((finding) => [finding.id, finding]));
  const shown: Sentence[] = [];
  const dropped: Sentence[] = [];
  for (const text of sentencesOf(reply)) {
    const cites = Array.from(text.matchAll(CITE)).flatMap((match) => match[1].split(/\s*[,;]\s*/));
    const bare = text.replace(CITE, " ");
    const cited = cites.map((id) => byId.get(id));
    let why: string | null = null;
    if (!cites.length) {
      why = "it names no finding";
    } else if (cited.some((finding) => finding === undefined)) {
      why = `it names ${cites.filter((id) => !byId.has(id)).join(", ")}, which the analysis did not find`;
    } else if (MEDICAL.test(bare)) {
      why = "it strays into injury or medicine";
    } else {
      const numbers = cited.flatMap((finding) => numbersOf(finding as Finding));
      const unheld = Array.from(bare.matchAll(NUMBER)).map((match) => match[0]).filter((token) => !holds(numbers, token));
      const unsure = cited.filter((finding) => (finding as Finding).call === TOO_CLOSE).map((finding) => (finding as Finding).id);
      if (unheld.length) why = `${unheld.join(", ")} is not in ${cites.join(", ")}`;
      else if (unsure.length && !UNSURE.test(bare)) why = `it states as fact what ${unsure.join(", ")} left too close to call`;
    }
    (why ? dropped : shown).push({ text, cites, why });
  }
  return { shown, dropped };
}

/** The coaching as the player sees it: the sentences that passed and a note of what was left out.
 * When none passed, the findings' own sentences about the rally and the player stand in. */
export function coachingText(checked: CheckedCoaching, findings: Finding[], player: "near" | "far"): string {
  const left = checked.dropped.length
    ? `\n\n(${checked.dropped.length} ${checked.dropped.length === 1 ? "sentence was" : "sentences were"} left out: the analysis does not support ${checked.dropped.length === 1 ? "it" : "them"}.)`
    : "";
  if (checked.shown.length) return checked.shown.map((sentence) => sentence.text).join(" ") + left;
  const own = findings.filter((finding) => finding.kind !== "shot" && (finding.kind !== "player" || finding.about === player));
  return own.map((finding) => `${finding.text} [${finding.id}]`).join(" ") + left;
}
