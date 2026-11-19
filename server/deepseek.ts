import { CoachingContext, buildCoachingPrompt } from "./analysis";
import { buildFindingsPrompt, checkCoaching, coachingText, findingsOf } from "./coaching";

type DeepSeekResponse = {
  choices?: Array<{ message?: { content?: string } }>;
  error?: { message?: string };
};

/** The prompt for a result: from its findings when it has them, from its metrics when it has none. */
export function coachingPrompt(context: CoachingContext): string {
  const findings = findingsOf(context.result);
  return findings ? buildFindingsPrompt(context.player, findings, context.question) : buildCoachingPrompt(context);
}

/** Coaching for a result: from its findings when it has them, with every sentence checked against
 * them; from its metrics as before when it has none. */
export async function requestDeepSeekCoaching(context: CoachingContext) {
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) throw new Error("DeepSeek coaching is not configured.");
  const findings = findingsOf(context.result);

  const response = await fetch("https://api.deepseek.com/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: "deepseek-v4-pro",
      messages: [{ role: "user", content: coachingPrompt(context) }],
      thinking: { type: "enabled" },
      reasoning_effort: "high",
      temperature: 0.2,
      stream: false,
    }),
  });

  const payload = await response.json() as DeepSeekResponse;
  if (!response.ok) throw new Error(payload.error?.message || "DeepSeek coaching request failed.");
  const feedback = payload.choices?.[0]?.message?.content?.trim();
  if (!feedback) throw new Error("DeepSeek returned no coaching feedback.");
  return findings ? coachingText(checkCoaching(feedback, findings), findings, context.player) : feedback;
}
