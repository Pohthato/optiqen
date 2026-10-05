import { COOKIE_NAME } from "@shared/const";
import { TRPCError } from "@trpc/server";
import { nanoid } from "nanoid";
import { z } from "zod";
import { AnalysisResult, CourtCorner, CourtType, buildCoachingPrompt, validateCalibration } from "./analysis";
import { analysisResultSchema, cornerSchema } from "./analysisSchema";
import { submitAnalysisWorkerJob, submitWorkerWarmup } from "./analysisWorker";
import { reconcileAnalysisWorkerJob } from "./analysisCompletion";
import { authRouter } from "./auth";
import { systemRouter } from "./_core/systemRouter";
import { protectedProcedure, publicProcedure, router } from "./_core/trpc";
import { createAnalysisSession, getAnalysisSessionForUser, listAnalysisSessionsForUser, updateAnalysisSessionForUser } from "./db";
import { requestDeepSeekCoaching } from "./deepseek";
import { storageCreateUploadUrl, storageGetSignedUrl } from "./storage";

const permittedVideoTypes = ["video/mp4", "video/quicktime", "video/webm"] as const;

function safeFilename(filename: string) {
  return filename.replace(/[^a-zA-Z0-9._-]/g, "_").slice(0, 180) || "source-video";
}

export const appRouter = router({
  system: systemRouter,
  auth: authRouter,
  upload: router({
    prepareVideo: protectedProcedure.input(z.object({
      filename: z.string().trim().min(1).max(255),
      mimeType: z.enum(permittedVideoTypes),
      bytes: z.number().int().positive().max(1_500_000_000),
    })).mutation(async ({ ctx, input }) => {
      const filename = safeFilename(input.filename);
      return storageCreateUploadUrl(
        `analysis-sources/${ctx.user.id}/${nanoid(12)}-${filename}`,
        input.mimeType,
      );
    }),
  }),
  analysis: router({
    list: protectedProcedure.query(({ ctx }) => listAnalysisSessionsForUser(ctx.user.id)),
    get: protectedProcedure.input(z.object({ id: z.string().min(6).max(28) })).query(async ({ ctx, input }) => {
      const session = await getAnalysisSessionForUser(input.id, ctx.user.id);
      if (!session) throw new TRPCError({ code: "NOT_FOUND", message: "Analysis session not found." });
      return session;
    }),
    createDraft: protectedProcedure.input(z.object({
      sourceName: z.string().trim().min(1).max(255),
      sourceStorageKey: z.string().trim().min(1).max(768),
      sourceDurationMs: z.number().int().positive().max(180_000).optional(),
      selectedPlayer: z.enum(["near", "far"]),
      courtType: z.enum(["singles", "doubles"]).default("singles"),
      corners: z.array(cornerSchema).min(3).max(4),
    })).mutation(async ({ ctx, input }) => {
      const calibration = validateCalibration(input.corners as CourtCorner[], input.courtType as CourtType);
      const id = nanoid(18);
      await createAnalysisSession({ id, userId: ctx.user.id, sourceName: input.sourceName, sourceStorageKey: input.sourceStorageKey, sourceDurationMs: input.sourceDurationMs, selectedPlayer: input.selectedPlayer, calibration });
      return { id, calibration };
    }),
    submit: protectedProcedure.input(z.object({
      id: z.string().min(6).max(28),
      requestedLayers: z.array(z.enum(["skeleton", "shuttle", "racket", "courtMap"])).min(1),
    })).mutation(async ({ ctx, input }) => {
      const session = await getAnalysisSessionForUser(input.id, ctx.user.id);
      if (!session) throw new TRPCError({ code: "NOT_FOUND", message: "Analysis session not found." });
      const calibration = session.calibration as ReturnType<typeof validateCalibration>;
      const sourceVideoUrl = await storageGetSignedUrl(session.sourceStorageKey);
      const courtType = (session.calibration as { courtType?: "singles" | "doubles" }).courtType ?? "singles";
      const worker = await submitAnalysisWorkerJob({ analysisId: session.id, videoStorageKey: session.sourceStorageKey, sourceVideoUrl, selectedPlayer: session.selectedPlayer, courtType, calibration, requestedLayers: input.requestedLayers });
      await updateAnalysisSessionForUser(session.id, ctx.user.id, { status: "queued", workerJobId: worker.jobId });
      return worker;
    }),
    warmupGpu: protectedProcedure.mutation(async () => {
      return submitWorkerWarmup();
    }),
    refresh: protectedProcedure.input(z.object({ id: z.string().min(6).max(28) })).mutation(async ({ ctx, input }) => {
      const session = await getAnalysisSessionForUser(input.id, ctx.user.id);
      if (!session) throw new TRPCError({ code: "NOT_FOUND", message: "Analysis session not found." });
      if (!session.workerJobId) return { status: session.status, updated: false };
      return reconcileAnalysisWorkerJob(session.id);
    }),
  }),
  coach: router({
    previewPrompt: protectedProcedure.input(z.object({ player: z.enum(["near", "far"]), result: analysisResultSchema, question: z.string().trim().max(1_200).optional() })).query(({ input }) => {
      return buildCoachingPrompt({ player: input.player, calibration: input.result.calibration, result: input.result as AnalysisResult, question: input.question });
    }),
    chat: protectedProcedure.input(z.object({ player: z.enum(["near", "far"]), result: analysisResultSchema, question: z.string().trim().min(1).max(1_200) })).mutation(async ({ input }) => {
      const result = input.result as AnalysisResult;
      const feedback = await requestDeepSeekCoaching({ player: input.player, calibration: result.calibration, result, question: input.question });
      return { feedback };
    }),
  }),
});

export type AppRouter = typeof appRouter;
