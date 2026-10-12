import { CoachingContext, buildCoachingMessages } from "./analysis";

type DeepSeekResponse = {
  choices?: Array<{ message?: { content?: string } }>;
  usage?: { prompt_cache_hit_tokens?: number; prompt_cache_miss_tokens?: number; completion_tokens?: number };
  error?: { message?: string };
};

export async function requestDeepSeekCoaching(context: CoachingContext) {
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) throw new Error("DeepSeek coaching is not configured.");

  // DeepSeek caches request prefixes on its own (there is no cache_control), so the
  // messages keep the unchanging rules and session evidence ahead of the question.
  // Thinking mode ignores temperature, so none is sent.
  const response = await fetch("https://api.deepseek.com/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: "deepseek-v4-pro",
      messages: buildCoachingMessages(context),
      thinking: { type: "enabled" },
      reasoning_effort: "high",
      stream: false,
    }),
  });

  const payload = await response.json() as DeepSeekResponse;
  if (!response.ok) throw new Error(payload.error?.message || "DeepSeek coaching request failed.");
  const feedback = payload.choices?.[0]?.message?.content?.trim();
  if (!feedback) throw new Error("DeepSeek returned no coaching feedback.");

  const usage = payload.usage;
  if (usage) {
    const hit = usage.prompt_cache_hit_tokens ?? 0;
    const miss = usage.prompt_cache_miss_tokens ?? 0;
    console.log(`[Coach] ${hit} of ${hit + miss} prompt tokens cached, ${usage.completion_tokens ?? 0} completion tokens.`);
  }
  return feedback;
}
