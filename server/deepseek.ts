import { CoachingContext, buildCoachingPrompt } from "./analysis";

type DeepSeekResponse = {
  choices?: Array<{ message?: { content?: string } }>;
  error?: { message?: string };
};

export async function requestDeepSeekCoaching(context: CoachingContext) {
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) throw new Error("DeepSeek coaching is not configured.");

  const response = await fetch("https://api.deepseek.com/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: "deepseek-v4-pro",
      messages: [{ role: "user", content: buildCoachingPrompt(context) }],
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
  return feedback;
}
