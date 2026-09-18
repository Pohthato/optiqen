import React from "react";
import {
  Activity,
  ArrowRight,
  Check,
  ChevronDown,
  CircleDot,
  Clock3,
  Crosshair,
  Eye,
  FileVideo,
  Footprints,
  Gauge,
  Info,
  Layers3,
  Lock,
  Maximize2,
  Menu,
  MousePointer2,
  Pause,
  Play,
  Plus,
  Radar,
  RotateCcw,
  Ruler,
  Scan,
  ShieldCheck,
  Sparkles,
  Target,
  UploadCloud,
  Video,
  WandSparkles,
  Workflow,
  X,
} from "lucide-react";
import { ChangeEvent, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { useAuth } from "@/_core/hooks/useAuth";
import { startLogin } from "@/const";
import { trpc } from "@/lib/trpc";
import { AIChatBox, Message } from "@/components/AIChatBox";
import {
  AnalysisOverlay,
  VerifiedOverlayFrame,
} from "@/components/AnalysisOverlay";
import { getCompletedOverlayFrames } from "@/lib/completedAnalysis";
import { getAuthAction, getAuthButtonLabel } from "@/lib/authUi";
import { AuthControl } from "@/components/AuthControl";
import {
  getCalibrationPointLabel,
  getEstimatedCornerLabel,
  getProvisionalFourthCorner,
  resolveCalibrationPoints,
} from "@/lib/calibration";
import { buildCalibrationPayload } from "@/lib/calibrationPayload";
import { getAnalysisQueueNotice } from "@/lib/analysisQueue";

type Corner = { x: number; y: number };
type Layer = "Skeleton" | "Shuttle" | "Racket" | "Court map";
type StoredAnalysisResult = {
  annotatedVideoStorageKey?: string;
  metrics?: Array<{
    metric: string;
    value: number;
    unit: string;
    confidence: number;
  }>;
  quality?: {
    usableFrameRatio?: number;
    poseTrackConfidence?: number;
    shuttleTrackConfidence?: number;
  };
  diagnostics?: {
    warnings?: string[];
  };
  overlays?: VerifiedOverlayFrame[];
  calibration?: {
    confidence?:
      | "unverified"
      | "provisional"
      | "validated"
      | "image_space_only";
    supportsCourtMapping?: boolean;
  };
};

const supportedLayers: Layer[] = ["Skeleton", "Shuttle", "Racket", "Court map"];
const allowedMimeTypes = [
  "video/mp4",
  "video/quicktime",
  "video/webm",
] as const;

function formatDuration(seconds: number) {
  if (!Number.isFinite(seconds) || seconds <= 0) return "Ready for metadata";
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${minutes}:${remainingSeconds}`;
}

function formatFileSize(bytes: number) {
  if (!bytes) return "—";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1
  );
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function SkeletonReference() {
  return (
    <svg
      viewBox="0 0 130 220"
      className="h-full w-full overflow-visible"
      aria-hidden="true"
    >
      <defs>
        <filter id="glow">
          <feGaussianBlur stdDeviation="1.5" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <g
        fill="none"
        stroke="#B7FA59"
        strokeLinecap="round"
        strokeLinejoin="round"
        filter="url(#glow)"
      >
        <path
          d="M69 32 L66 76 L41 103 M66 76 L88 99 M55 90 L79 89 M66 76 L55 130 L42 186 M55 130 L80 185"
          strokeWidth="4"
        />
        <path d="M42 186 L24 198 M80 185 L102 197" strokeWidth="4" />
        <path d="M88 99 L112 74" stroke="#78E7F4" strokeWidth="3" />
      </g>
      <circle
        cx="69"
        cy="20"
        r="13"
        fill="#0D181D"
        stroke="#B7FA59"
        strokeWidth="4"
      />
      {[
        [69, 32],
        [66, 76],
        [41, 103],
        [88, 99],
        [55, 130],
        [42, 186],
        [80, 185],
        [112, 74],
      ].map(([cx, cy]) => (
        <circle
          key={`${cx}-${cy}`}
          cx={cx}
          cy={cy}
          r="4.5"
          fill="#F2FFE0"
          stroke="#B7FA59"
          strokeWidth="2"
        />
      ))}
      <circle cx="119" cy="67" r="3.5" fill="#FFDE73" />
    </svg>
  );
}

function CourtBlueprint({ compact = false }: { compact?: boolean }) {
  return (
    <svg
      viewBox="0 0 420 252"
      className="h-full w-full"
      aria-label="Badminton court map"
    >
      <rect
        x="20"
        y="12"
        width="380"
        height="228"
        rx="2"
        fill="rgba(183,250,89,.045)"
        stroke="rgba(183,250,89,.72)"
        strokeWidth="2"
      />
      <path
        d="M20 60 H400 M20 192 H400 M72 12 V240 M348 12 V240 M210 12 V240"
        fill="none"
        stroke="rgba(183,250,89,.55)"
        strokeWidth="1.5"
      />
      <path d="M20 126 H400" stroke="rgba(242,255,224,.9)" strokeWidth="2" />
      <path d="M204 118 H216 V134 H204z" fill="#B7FA59" />
      <circle cx="155" cy="168" r="5" fill="#78E7F4" />
      <circle cx="278" cy="81" r="5" fill="#FFDE73" />
      {!compact && (
        <>
          <path
            d="M155 168 C182 139 221 130 278 81"
            fill="none"
            stroke="#FFDE73"
            strokeDasharray="5 5"
            strokeWidth="2"
          />
          <path
            d="M155 168 C170 188 201 203 242 205"
            fill="none"
            stroke="#78E7F4"
            strokeDasharray="5 5"
            strokeWidth="2"
          />
        </>
      )}
    </svg>
  );
}

export default function Home() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const { isAuthenticated, logout } = useAuth();
  const uploadMutation = trpc.upload.prepareVideo.useMutation();
  const draftMutation = trpc.analysis.createDraft.useMutation();
  const submitMutation = trpc.analysis.submit.useMutation();
  const warmupGpuMutation = trpc.analysis.warmupGpu.useMutation();
  const refreshMutation = trpc.analysis.refresh.useMutation();
  const coachMutation = trpc.coach.chat.useMutation();
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [calibrationMode, setCalibrationMode] = useState(false);
  const [corners, setCorners] = useState<Corner[]>([]);
  const [selectedPlayer, setSelectedPlayer] = useState<"near" | "far">("near");
  const [layers, setLayers] = useState<Record<Layer, boolean>>({
    Skeleton: true,
    Shuttle: true,
    Racket: true,
    "Court map": true,
  });
  const [analysisRequested, setAnalysisRequested] = useState(false);
  const [submissionState, setSubmissionState] = useState<
    "idle" | "uploading" | "queued" | "error"
  >("idle");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [coachMessages, setCoachMessages] = useState<Message[]>([]);
  const [profileMenuOpen, setProfileMenuOpen] = useState(false);
  const sessionQuery = trpc.analysis.get.useQuery(
    { id: sessionId ?? "pending" },
    { enabled: Boolean(sessionId), refetchInterval: sessionId ? 12_000 : false }
  );
  const workerStatus = sessionQuery.data?.status;

  useEffect(() => {
    if (!sessionId || (workerStatus !== "queued" && workerStatus !== "processing")) return;
    const refresh = () => {
      refreshMutation.mutate(
        { id: sessionId },
        { onSuccess: () => void sessionQuery.refetch() }
      );
    };
    refresh();
    const timer = window.setInterval(refresh, 12_000);
    return () => window.clearInterval(timer);
  }, [sessionId, workerStatus]);
  const queueNotice = getAnalysisQueueNotice(
    sessionQuery.data?.status ?? "draft",
    sessionQuery.data?.createdAt
  );

  useEffect(
    () => () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl);
    },
    [videoUrl]
  );

  const calibrationStatus =
    corners.length >= 4
      ? "Validated geometry"
      : corners.length >= 3
        ? "3-point provisional"
        : "Awaiting points";
  const resolvedCorners = resolveCalibrationPoints(corners);
  const completedCalibration =
    sessionQuery.data?.status === "completed"
      ? (sessionQuery.data.result as StoredAnalysisResult | null)?.calibration
      : undefined;
  const displayCalibrationStatus =
    completedCalibration?.confidence === "image_space_only"
      ? "Image-space only"
      : completedCalibration?.confidence === "validated"
        ? "Validated geometry"
        : calibrationStatus;
  const estimatedCorner = getProvisionalFourthCorner(corners);
  const canStartAnalysis = Boolean(videoFile && corners.length >= 3);
  const hasOAuthConfig = Boolean(
    import.meta.env.VITE_OAUTH_PORTAL_URL && import.meta.env.VITE_APP_ID
  );
  const authAction = getAuthAction(isAuthenticated, hasOAuthConfig);
  const authButtonLabel = getAuthButtonLabel(authAction);
  const playbackProgress = duration ? (currentTime / duration) * 100 : 0;
  const provenance = useMemo(
    () =>
      videoFile
        ? `${videoFile.name} · local preview`
        : "No source footage selected",
    [videoFile]
  );
  const completedResult =
    sessionQuery.data?.status === "completed"
      ? (sessionQuery.data.result as StoredAnalysisResult | null)
      : null;
  const annotatedVideoUrl = completedResult?.annotatedVideoStorageKey
    ? `/api/local-storage/${completedResult.annotatedVideoStorageKey}`
    : null;
  const verifiedOverlayFrames = getCompletedOverlayFrames({
    status: sessionQuery.data?.status,
    result: sessionQuery.data?.result as
      | StoredAnalysisResult
      | null
      | undefined,
  });
  const displayedVideoUrl = annotatedVideoUrl ?? videoUrl;
  const hasCompletedResult =
    sessionQuery.data?.status === "completed" &&
    Boolean(sessionQuery.data?.result);

  const selectVideo = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !file.type.startsWith("video/")) return;
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoUrl(URL.createObjectURL(file));
    setVideoFile(file);
    setDuration(0);
    setCurrentTime(0);
    setCorners([]);
    setCalibrationMode(true);
    setAnalysisRequested(false);
    setSubmissionState("idle");
    setSessionId(null);
    if (isAuthenticated) {
      // Predictive GPU warmup: the user is about to calibrate and submit, so
      // start the serverless worker booting now. Errors are non-fatal.
      void warmupGpuMutation
        .mutateAsync()
        .then(result => {
          if (result.accepted) {
            toast.info("GPU worker is warming up for your analysis.");
          }
        })
        .catch(() => {
          /* cold-start prefetch is best-effort */
        });
    }
  };

  const recordCorner = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!calibrationMode || corners.length >= 4 || !stageRef.current) return;
    const box = stageRef.current.getBoundingClientRect();
    const point = {
      x: Math.max(
        5,
        Math.min(95, ((event.clientX - box.left) / box.width) * 100)
      ),
      y: Math.max(
        5,
        Math.min(95, ((event.clientY - box.top) / box.height) * 100)
      ),
    };
    setCorners(previous => [...previous, point]);
  };

  const handleAuthClick = () => {
    if (authAction === "profile") {
      setProfileMenuOpen(open => !open);
      return;
    }
    if (authAction === "sign_in") {
      startLogin();
      return;
    }
    toast.info(
      "Local sign-in is not configured. Add VITE_OAUTH_PORTAL_URL and VITE_APP_ID to .env.local, or open the hosted Netoval app to sign in."
    );
  };

  const handleLogout = async () => {
    await logout();
    setProfileMenuOpen(false);
  };

  const togglePlayback = async () => {
    if (!videoRef.current) return;
    if (videoRef.current.paused) {
      await videoRef.current.play();
    } else {
      videoRef.current.pause();
    }
  };

  const scrub = (event: ChangeEvent<HTMLInputElement>) => {
    const nextTime = Number(event.target.value);
    setCurrentTime(nextTime);
    if (videoRef.current) videoRef.current.currentTime = nextTime;
  };

  const submitAnalysis = async () => {
    if (!isAuthenticated) {
      if (hasOAuthConfig) {
        toast.info(
          "Sign in to protect the video and start a private analysis session."
        );
        startLogin();
      } else {
        toast.info(
          "Local sign-in is not configured. Add VITE_OAUTH_PORTAL_URL and VITE_APP_ID to .env.local, or open the hosted Netoval app to sign in."
        );
      }
      return;
    }
    if (!videoFile || corners.length < 3) return;
    if (!(allowedMimeTypes as readonly string[]).includes(videoFile.type)) {
      toast.error(
        "Use an MP4, MOV, or WebM video with a recognized media type."
      );
      return;
    }
    if (videoFile.size > 1_500_000_000) {
      toast.error("Use a video below 1.5 GB. Trim a longer match into rallies before analysis.");
      return;
    }
    if (duration > 600) {
      toast.error("Use footage up to 10 minutes. For longer matches, upload individual rallies or drills.");
      return;
    }

    setSubmissionState("uploading");
    let submissionStage = "secure upload preparation";
    try {
      const artifact = await uploadMutation.mutateAsync({
        filename: videoFile.name,
        mimeType: videoFile.type as (typeof allowedMimeTypes)[number],
        bytes: videoFile.size,
      });
      submissionStage = "video transfer";
      const upload = await fetch(artifact.uploadUrl, {
        method: "PUT",
        headers: { "Content-Type": artifact.contentType },
        body: videoFile,
      });
      if (!upload.ok)
        throw new Error(`Video transfer returned HTTP ${upload.status}.`);

      submissionStage = "analysis session record";
      const draft = await draftMutation.mutateAsync({
        sourceName: videoFile.name,
        sourceStorageKey: artifact.key,
        sourceDurationMs:
          duration > 0 ? Math.round(duration * 1_000) : undefined,
        selectedPlayer,
        corners: buildCalibrationPayload(corners),
      });
      const requestedLayers = supportedLayers
        .filter(layer => layers[layer])
        .map(
          layer =>
            ({
              Skeleton: "skeleton",
              Shuttle: "shuttle",
              Racket: "racket",
              "Court map": "courtMap",
            })[layer]
        ) as Array<"skeleton" | "shuttle" | "racket" | "courtMap">;
      submissionStage = "computer-vision worker handoff";
      await submitMutation.mutateAsync({ id: draft.id, requestedLayers });
      setSessionId(draft.id);
      setAnalysisRequested(true);
      setSubmissionState("queued");
      toast.success(
        "Private video analysis was accepted by the compute worker."
      );
    } catch (error) {
      setSubmissionState("error");
      const detail =
        error instanceof Error ? error.message : "Unknown submission error.";
      toast.error(`${submissionStage} failed: ${detail}`);
    }
  };

  const refreshAnalysis = async () => {
    if (!sessionId) return;
    try {
      const update = await refreshMutation.mutateAsync({ id: sessionId });
      await sessionQuery.refetch();
      if (update.status === "completed")
        toast.success("Verified annotated replay is ready.");
      else if (update.status === "failed")
        toast.error(("error" in update ? update.error : undefined) || "The worker reported an analysis failure.");
      else toast.info(`Analysis is ${update.status}.`);
    } catch (error) {
      toast.error(
        error instanceof Error
          ? error.message
          : "Unable to check analysis status."
      );
    }
  };

  const askCoach = async (question: string) => {
    if (!sessionQuery.data?.result) return;
    setCoachMessages(messages => [
      ...messages,
      { role: "user", content: question },
    ]);
    try {
      const reply = await coachMutation.mutateAsync({
        player: selectedPlayer,
        result: sessionQuery.data.result as never,
        question,
      });
      setCoachMessages(messages => [
        ...messages,
        { role: "assistant", content: reply.feedback },
      ]);
    } catch (error) {
      const explanation =
        error instanceof Error
          ? error.message
          : "Coaching feedback could not be generated.";
      setCoachMessages(messages => [
        ...messages,
        {
          role: "assistant",
          content: `I could not complete that coaching request: ${explanation}`,
        },
      ]);
    }
  };

  return (
    <div className="min-h-screen overflow-hidden bg-[#0b1116] text-slate-100">
      <header className="relative z-20 flex h-[74px] items-center justify-between border-b border-white/10 bg-[#0d151b]/95 px-5 backdrop-blur-xl lg:px-8">
        <div className="flex items-center gap-6">
          <button
            className="rounded-lg p-2 text-slate-400 transition-colors hover:bg-white/5 hover:text-white lg:hidden"
            aria-label="Open navigation"
          >
            <Menu className="h-5 w-5" />
          </button>
          <div className="flex items-center gap-3">
            <div className="relative flex h-9 w-9 items-center justify-center rounded-xl bg-[#b7fa59] text-[#0c1717] shadow-[0_0_28px_rgba(183,250,89,.22)]">
              <Radar className="h-5 w-5" />
              <span className="absolute -right-1 -top-1 h-2.5 w-2.5 rounded-full border-2 border-[#0d151b] bg-[#78e7f4]" />
            </div>
            <div>
              <p className="text-[15px] font-extrabold tracking-[-0.045em] text-white">
                NETOVAL
              </p>
              <p className="font-mono-data text-[9px] tracking-[0.18em] text-[#b7fa59]">
                PERFORMANCE LAB
              </p>
            </div>
          </div>
          <div className="hidden h-6 w-px bg-white/10 md:block" />
          <button className="hidden items-center gap-2 text-sm font-medium text-slate-400 transition-colors hover:text-white md:flex">
            Sessions <ChevronDown className="h-4 w-4" />
          </button>
        </div>
        <AuthControl
          action={authAction}
          label={authButtonLabel}
          profileMenuOpen={profileMenuOpen}
          onAuthClick={handleAuthClick}
          onLogout={handleLogout}
        />
      </header>

      <main className="relative mx-auto max-w-[1700px] px-4 pb-9 pt-6 md:px-6 lg:px-8">
        <div className="pointer-events-none absolute inset-x-0 top-0 h-[420px] bg-[radial-gradient(ellipse_at_50%_-5%,rgba(70,103,94,.22),transparent_58%)]" />
        <div className="relative mb-6 flex flex-col justify-between gap-4 xl:flex-row xl:items-end">
          <div>
            <div className="mb-3 flex items-center gap-2 font-mono-data text-[10px] uppercase tracking-[0.16em] text-[#91a4aa]">
              <span>Workspace</span>
              <span className="text-slate-600">/</span>
              <span className="text-[#b7fa59]">New analysis</span>
            </div>
            <h1 className="max-w-3xl text-3xl font-extrabold tracking-[-0.055em] text-white md:text-[42px]">
              See the movement beneath the rally.
            </h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">
              Calibrate the court once, preserve the real camera perspective,
              then build a frame-by-frame evidence trail for movement, timing,
              technique, and tactical recovery.
            </p>
          </div>
          <div className="flex items-center gap-2 self-start rounded-xl border border-white/8 bg-[#121c22]/85 p-1.5 xl:self-auto">
            {[
              { icon: UploadCloud, label: "1 · Intake", active: !videoFile },
              {
                icon: Crosshair,
                label: "2 · Calibrate",
                active: Boolean(videoFile) && corners.length < 3,
              },
              {
                icon: Activity,
                label: "3 · Analyse",
                active: canStartAnalysis,
              },
            ].map(({ icon: Icon, label, active }) => (
              <div
                key={label}
                className={`flex items-center gap-2 rounded-lg px-3 py-2 text-xs font-semibold ${active ? "bg-white/8 text-white" : "text-slate-500"}`}
              >
                <Icon
                  className={`h-3.5 w-3.5 ${active ? "text-[#b7fa59]" : ""}`}
                />
                {label}
              </div>
            ))}
          </div>
        </div>

        <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_328px]">
          <section className="panel-surface overflow-hidden rounded-2xl">
            <div className="flex flex-col gap-4 border-b border-white/8 px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-center gap-3">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#b7fa59]/10 text-[#b7fa59]">
                  <Video className="h-4 w-4" />
                </div>
                <div>
                  <h2 className="text-sm font-bold tracking-[-0.02em] text-white">
                    Analysis canvas
                  </h2>
                  <p className="font-mono-data text-[10px] text-slate-500">
                    {provenance}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span
                  className={`rounded-full border px-2.5 py-1 font-mono-data text-[10px] ${corners.length >= 3 ? "border-[#b7fa59]/30 bg-[#b7fa59]/10 text-[#cffe91]" : "border-white/10 bg-white/[.035] text-slate-400"}`}
                >
                  {displayCalibrationStatus}
                </span>
                <button
                  onClick={() => setCalibrationMode(enabled => !enabled)}
                  className={`flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs font-bold transition-all ${calibrationMode ? "border-[#b7fa59]/50 bg-[#b7fa59] text-[#102018]" : "border-white/10 bg-white/5 text-slate-200 hover:border-white/25"}`}
                >
                  <MousePointer2 className="h-3.5 w-3.5" />
                  {calibrationMode ? "Picking corners" : "Calibrate"}
                </button>
              </div>
            </div>

            <div
              ref={stageRef}
              onClick={recordCorner}
              className={`court-grid relative aspect-video min-h-[310px] overflow-hidden bg-[#101c1d] ${calibrationMode && corners.length < 4 ? "cursor-crosshair" : ""}`}
            >
              {displayedVideoUrl ? (
                <video
                  key={displayedVideoUrl}
                  ref={videoRef}
                  src={displayedVideoUrl}
                  className="h-full w-full object-contain"
                  onLoadedMetadata={event =>
                    setDuration(event.currentTarget.duration)
                  }
                  onTimeUpdate={event =>
                    setCurrentTime(event.currentTarget.currentTime)
                  }
                  onPlay={() => setIsPlaying(true)}
                  onPause={() => setIsPlaying(false)}
                  playsInline
                />
              ) : (
                <>
                  <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_50%_110%,rgba(183,250,89,.13),transparent_55%),linear-gradient(128deg,rgba(14,49,47,.76),rgba(10,19,24,.92)_52%,rgba(25,38,38,.92))]" />
                  <div className="absolute left-1/2 top-1/2 h-[78%] w-[59%] -translate-x-1/2 -translate-y-1/2 rotate-[-3deg] border-2 border-[#b7fa59]/80 shadow-[0_0_0_30px_rgba(22,72,65,.14),0_0_90px_rgba(183,250,89,.08)]">
                    <div className="absolute inset-x-[13%] top-1/2 border-t-2 border-[#efffd4]/70" />
                    <div className="absolute inset-y-0 left-1/2 border-l border-[#b7fa59]/40" />
                    <div className="absolute inset-x-[13%] top-[21%] border-t border-[#b7fa59]/35" />
                    <div className="absolute inset-x-[13%] bottom-[21%] border-t border-[#b7fa59]/35" />
                  </div>
                  <div className="absolute bottom-[13%] left-[34%] h-[37%] w-[19%] opacity-90">
                    <SkeletonReference />
                  </div>
                  <div className="absolute right-[13%] top-[15%] flex items-center gap-2 rounded-lg border border-white/10 bg-[#0a1318]/75 px-3 py-2 backdrop-blur">
                    <Scan className="h-4 w-4 text-[#b7fa59]" />
                    <span className="font-mono-data text-[10px] uppercase tracking-[.14em] text-slate-300">
                      Upload to calibrate
                    </span>
                  </div>
                </>
              )}

              <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(90deg,transparent_49.9%,rgba(183,250,89,.08)_50%,transparent_50.1%)] opacity-40" />
              {resolvedCorners.length > 1 && (
                <svg
                  className="pointer-events-none absolute inset-0 h-full w-full"
                  viewBox="0 0 100 100"
                  preserveAspectRatio="none"
                >
                  <polyline
                    points={resolvedCorners
                      .map(point => `${point.x},${point.y}`)
                      .join(" ")}
                    fill="none"
                    stroke="#b7fa59"
                    strokeWidth="0.42"
                    strokeDasharray={corners.length === 4 ? "0" : "1.2 1.2"}
                  />
                </svg>
              )}
              {corners.map((corner, index) => (
                <div
                  key={`${corner.x}-${corner.y}-${index}`}
                  className="pointer-events-none absolute -translate-x-1/2 -translate-y-1/2"
                  style={{ left: `${corner.x}%`, top: `${corner.y}%` }}
                >
                  <div className="flex h-7 w-7 items-center justify-center rounded-full border-2 border-[#efffd4] bg-[#b7fa59] font-mono-data text-[11px] font-bold text-[#102018] shadow-[0_0_20px_rgba(183,250,89,.5)]">
                    {index + 1}
                  </div>
                  <span className="absolute left-8 top-1 whitespace-nowrap rounded bg-[#071015]/90 px-1.5 py-0.5 font-mono-data text-[9px] text-[#d9ffad]">
                    {getCalibrationPointLabel(index)}
                  </span>
                </div>
              ))}
              {estimatedCorner && (
                <div
                  className="pointer-events-none absolute -translate-x-1/2 -translate-y-1/2 opacity-80"
                  style={{
                    left: `${estimatedCorner.x}%`,
                    top: `${estimatedCorner.y}%`,
                  }}
                >
                  <div className="flex h-7 w-7 items-center justify-center rounded-full border border-dashed border-[#ffde73] bg-[#ffde73]/10 font-mono-data text-[11px] font-bold text-[#ffde73]">
                    4
                  </div>
                  <span className="absolute left-8 top-1 whitespace-nowrap rounded bg-[#071015]/90 px-1.5 py-0.5 font-mono-data text-[9px] text-[#ffde73]">
                    {getEstimatedCornerLabel()}
                  </span>
                </div>
              )}
              {!annotatedVideoUrl && verifiedOverlayFrames.length > 0 && (
                <AnalysisOverlay
                  frames={verifiedOverlayFrames}
                  currentTimeMs={currentTime * 1_000}
                  layers={layers}
                />
              )}
              {calibrationMode && corners.length < 4 && (
                <div className="pointer-events-none absolute inset-x-4 bottom-4 flex items-center justify-between rounded-xl border border-[#b7fa59]/20 bg-[#0c171a]/85 px-3 py-2.5 backdrop-blur">
                  <div className="flex items-center gap-2">
                    <CircleDot className="h-4 w-4 text-[#b7fa59]" />
                    <p className="text-xs text-slate-200">
                      {corners.length >= 3 ? (
                        <>
                          Optional: add{" "}
                          <span className="font-bold text-[#d9ffad]">
                            {corners.length >= 3
                              ? "a fourth visible corner"
                              : "any visible corner"}
                          </span>{" "}
                          to improve geometry confidence.
                        </>
                      ) : (
                        <>
                          Click{" "}
                          <span className="font-bold text-[#d9ffad]">
                            {corners.length >= 3
                              ? "a fourth visible corner"
                              : "any visible corner"}
                          </span>{" "}
                          at the actual line intersection.
                        </>
                      )}
                    </p>
                  </div>
                  <span className="font-mono-data text-[10px] text-slate-500">
                    {corners.length}/3 min
                  </span>
                </div>
              )}
              {videoUrl && !calibrationMode && !analysisRequested && (
                <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-[#081015]/25">
                  <div className="rounded-xl border border-white/10 bg-[#091216]/85 px-4 py-3 text-center backdrop-blur">
                    <p className="text-xs font-bold text-white">
                      Source preview
                    </p>
                    <p className="mt-1 max-w-[280px] text-[11px] leading-4 text-slate-400">
                      {displayCalibrationStatus === "Image-space only"
                        ? "Court geometry was not reliable enough for metre claims; pose and image-space movement remain available."
                        : "Frame annotations appear here after the compute worker returns verified tracks."}
                    </p>
                  </div>
                </div>
              )}
              {annotatedVideoUrl && (
                <div className="pointer-events-none absolute right-4 top-4 flex items-center gap-2 rounded-lg border border-[#b7fa59]/30 bg-[#0a1518]/85 px-3 py-2 backdrop-blur">
                  <Check className="h-3.5 w-3.5 text-[#b7fa59]" />
                  <span className="font-mono-data text-[9px] uppercase tracking-[.14em] text-[#d9ffad]">
                    Verified annotated replay
                  </span>
                </div>
              )}
            </div>

            <div className="border-t border-white/8 bg-[#0d171b]/75 px-4 py-3">
              <div className="flex flex-wrap items-center gap-x-4 gap-y-3">
                <button
                  onClick={togglePlayback}
                  disabled={!videoUrl}
                  className="flex h-9 w-9 items-center justify-center rounded-full bg-[#b7fa59] text-[#102018] transition-transform active:scale-95 disabled:cursor-not-allowed disabled:opacity-30"
                >
                  {isPlaying ? (
                    <Pause className="h-4 w-4" />
                  ) : (
                    <Play className="ml-0.5 h-4 w-4" />
                  )}
                </button>
                <span className="font-mono-data text-[11px] text-slate-300">
                  {formatDuration(currentTime)}{" "}
                  <span className="text-slate-600">/</span>{" "}
                  {formatDuration(duration)}
                </span>
                <input
                  aria-label="Video timeline"
                  type="range"
                  min="0"
                  max={duration || 1}
                  step="0.01"
                  value={currentTime}
                  onChange={scrub}
                  className="h-1 min-w-[180px] flex-1 cursor-pointer accent-[#b7fa59]"
                  style={{
                    background: `linear-gradient(90deg, #b7fa59 ${playbackProgress}%, rgba(255,255,255,.13) ${playbackProgress}%)`,
                  }}
                />
                <div className="flex items-center gap-2">
                  <button
                    className="rounded-md p-1.5 text-slate-400 hover:bg-white/5 hover:text-white"
                    aria-label="Toggle fullscreen"
                  >
                    <Maximize2 className="h-4 w-4" />
                  </button>
                  <span className="font-mono-data text-[10px] text-slate-500">
                    1.00×
                  </span>
                </div>
              </div>
            </div>
          </section>

          <aside className="space-y-5">
            <section className="panel-surface rounded-2xl p-5">
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <p className="font-mono-data text-[10px] uppercase tracking-[.16em] text-[#b7fa59]">
                    01 / source
                  </p>
                  <h2 className="mt-1 text-sm font-bold text-white">
                    Footage intake
                  </h2>
                </div>
                <FileVideo className="h-5 w-5 text-slate-500" />
              </div>
              {videoFile ? (
                <div className="rounded-xl border border-[#b7fa59]/20 bg-[#b7fa59]/5 p-3">
                  <div className="flex items-start gap-3">
                    <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[#b7fa59]/10 text-[#b7fa59]">
                      <FileVideo className="h-4 w-4" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-xs font-bold text-white">
                        {videoFile.name}
                      </p>
                      <p className="mt-0.5 font-mono-data text-[10px] text-slate-400">
                        {formatFileSize(videoFile.size)} ·{" "}
                        {formatDuration(duration)}
                      </p>
                    </div>
                    <button
                      onClick={() => {
                        if (videoUrl) URL.revokeObjectURL(videoUrl);
                        setVideoUrl(null);
                        setVideoFile(null);
                        setCorners([]);
                        setCalibrationMode(false);
                        setSubmissionState("idle");
                        setSessionId(null);
                      }}
                      className="rounded p-1 text-slate-500 hover:bg-white/5 hover:text-white"
                      aria-label="Remove selected video"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>
              ) : (
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="group flex w-full flex-col items-center rounded-xl border border-dashed border-white/15 bg-white/[.025] px-4 py-5 text-center transition-colors hover:border-[#b7fa59]/50 hover:bg-[#b7fa59]/5"
                >
                  <span className="mb-2 flex h-10 w-10 items-center justify-center rounded-xl border border-white/8 bg-[#111e23] text-[#b7fa59] group-hover:scale-105">
                    <UploadCloud className="h-5 w-5" />
                  </span>
                  <span className="text-xs font-bold text-white">
                    Add a rally or drill
                  </span>
                  <span className="mt-1 text-[11px] leading-4 text-slate-500">
                    MP4, MOV, WebM · stable side or rear court view
                  </span>
                </button>
              )}
              <input
                ref={fileInputRef}
                onChange={selectVideo}
                type="file"
                accept="video/mp4,video/quicktime,video/webm"
                className="hidden"
              />
              {videoFile && (
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="mt-3 flex w-full items-center justify-center gap-2 rounded-lg border border-white/10 py-2 text-xs font-semibold text-slate-300 transition-colors hover:bg-white/5"
                >
                  <Plus className="h-3.5 w-3.5" />
                  Replace source
                </button>
              )}
              <div className="mt-4 flex gap-2 rounded-lg border border-white/6 bg-[#090f14]/60 px-3 py-2.5">
                <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-[#78e7f4]" />
                <p className="text-[10.5px] leading-4 text-slate-500">
                  Keep the full court visible, avoid digital zoom, and use 60
                  fps or more when analysing fast overheads.
                </p>
              </div>
            </section>

            <section className="panel-surface rounded-2xl p-5">
              <div className="mb-4 flex items-start justify-between">
                <div>
                  <p className="font-mono-data text-[10px] uppercase tracking-[.16em] text-[#b7fa59]">
                    02 / geometry
                  </p>
                  <h2 className="mt-1 text-sm font-bold text-white">
                    Court calibration
                  </h2>
                </div>
                <Ruler className="h-5 w-5 text-slate-500" />
              </div>
              <div className="space-y-2">
                {corners.map((point, index) => (
                  <div
                    key={`${point.x}-${point.y}-${index}`}
                    className="flex items-center justify-between rounded-lg bg-[#b7fa59]/7 px-3 py-2.5"
                  >
                    <div className="flex items-center gap-2.5">
                      <span className="flex h-5 w-5 items-center justify-center rounded-full border border-[#b7fa59] bg-[#b7fa59] font-mono-data text-[9px] text-[#102018]">
                        {index + 1}
                      </span>
                      <span className="text-xs text-slate-200">
                        Observed point {index + 1}
                      </span>
                    </div>
                    <span className="font-mono-data text-[9px] text-slate-600">
                      {point.x.toFixed(0)}, {point.y.toFixed(0)}
                    </span>
                  </div>
                ))}
                {estimatedCorner && (
                  <div className="flex items-center justify-between rounded-lg bg-[#ffde73]/[.06] px-3 py-2.5">
                    <div className="flex items-center gap-2.5">
                      <span className="flex h-5 w-5 items-center justify-center rounded-full border border-dashed border-[#ffde73] font-mono-data text-[9px] text-[#ffde73]">
                        ~
                      </span>
                      <span className="text-xs text-[#ffde73]">
                        Fourth point · estimated
                      </span>
                    </div>
                    <span className="font-mono-data text-[9px] text-slate-600">
                      {estimatedCorner.x.toFixed(0)},{" "}
                      {estimatedCorner.y.toFixed(0)}
                    </span>
                  </div>
                )}
              </div>
              <div className="mt-3 flex items-start gap-2 text-[10.5px] leading-4 text-slate-500">
                <Lock className="mt-0.5 h-3.5 w-3.5 shrink-0 text-slate-600" />
                <p>
                  Click any three visible court intersections in any order. They
                  remain neutral observed points until the geometry solver
                  validates which court corner is missing; the fourth point is
                  shown as an estimate only.
                </p>
              </div>
              <button
                onClick={() => {
                  setCorners([]);
                  setCalibrationMode(Boolean(videoFile));
                }}
                disabled={!corners.length}
                className="mt-4 flex w-full items-center justify-center gap-2 rounded-lg border border-white/10 py-2 text-xs font-semibold text-slate-300 transition-colors hover:bg-white/5 disabled:cursor-not-allowed disabled:opacity-35"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                Reset points
              </button>
            </section>
          </aside>
        </div>

        <div className="mt-5 grid gap-5 lg:grid-cols-[minmax(0,1.3fr)_minmax(300px,.7fr)]">
          <section className="panel-surface rounded-2xl p-5">
            <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-start">
              <div>
                <p className="font-mono-data text-[10px] uppercase tracking-[.16em] text-[#b7fa59]">
                  03 / tracking brief
                </p>
                <h2 className="mt-1 text-lg font-extrabold tracking-[-.035em] text-white">
                  Tell the system what to resolve.
                </h2>
                <p className="mt-1 max-w-xl text-xs leading-5 text-slate-400">
                  Your original video remains the reference. Visual layers are
                  drawn from verified frame tracks; there are no persistent
                  player boxes in the review experience.
                </p>
              </div>
              <div className="flex rounded-lg border border-white/10 bg-[#0c151a] p-1">
                <button
                  onClick={() => setSelectedPlayer("near")}
                  className={`rounded-md px-3 py-1.5 text-xs font-bold transition-colors ${selectedPlayer === "near" ? "bg-white/10 text-white" : "text-slate-500"}`}
                >
                  Near player
                </button>
                <button
                  onClick={() => setSelectedPlayer("far")}
                  className={`rounded-md px-3 py-1.5 text-xs font-bold transition-colors ${selectedPlayer === "far" ? "bg-white/10 text-white" : "text-slate-500"}`}
                >
                  Far player
                </button>
              </div>
            </div>
            <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
              {supportedLayers.map(layer => {
                const icons = {
                  Skeleton: Activity,
                  Shuttle: CircleDot,
                  Racket: WandSparkles,
                  "Court map": Footprints,
                };
                const Icon = icons[layer];
                return (
                  <button
                    key={layer}
                    onClick={() =>
                      setLayers(previous => ({
                        ...previous,
                        [layer]: !previous[layer],
                      }))
                    }
                    className={`flex items-center gap-3 rounded-xl border p-3 text-left transition-all ${layers[layer] ? "border-[#b7fa59]/35 bg-[#b7fa59]/7" : "border-white/8 bg-white/[.02] opacity-55"}`}
                  >
                    <span
                      className={`flex h-8 w-8 items-center justify-center rounded-lg ${layers[layer] ? "bg-[#b7fa59]/12 text-[#b7fa59]" : "bg-white/5 text-slate-500"}`}
                    >
                      <Icon className="h-4 w-4" />
                    </span>
                    <span>
                      <span className="block text-xs font-bold text-slate-200">
                        {layer}
                      </span>
                      <span className="mt-0.5 block font-mono-data text-[9px] uppercase text-slate-500">
                        {layers[layer] ? "Resolve" : "Excluded"}
                      </span>
                    </span>
                  </button>
                );
              })}
            </div>
            <div className="mt-5 grid gap-3 border-t border-white/8 pt-5 md:grid-cols-[1fr_auto] md:items-center">
              <div className="flex items-start gap-3">
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[#78e7f4]/10 text-[#78e7f4]">
                  <Workflow className="h-4 w-4" />
                </div>
                <p className="text-[11px] leading-5 text-slate-400">
                  The production worker checks pose confidence, court
                  reprojection error, and shuttle continuity before it makes a
                  coaching inference. Unsupported frames stay visible but are
                  excluded from metric summaries.
                </p>
              </div>
              <button
                onClick={submitAnalysis}
                disabled={
                  !canStartAnalysis ||
                  submissionState === "uploading" ||
                  submissionState === "queued"
                }
                className="flex items-center justify-center gap-2 rounded-xl bg-[#b7fa59] px-4 py-3 text-xs font-extrabold text-[#102018] transition-all hover:bg-[#ceff7d] active:scale-[.98] disabled:cursor-not-allowed disabled:bg-white/10 disabled:text-slate-500"
              >
                <Sparkles className="h-4 w-4" />
                {submissionState === "uploading"
                  ? "Submitting securely..."
                  : submissionState === "queued"
                    ? "Queued for analysis"
                    : "Secure upload & analyse"}
                <ArrowRight className="h-4 w-4" />
              </button>
            </div>
            {analysisRequested && (
              <div
                className={`mt-4 flex items-start gap-3 rounded-xl border p-3 ${queueNotice.delayed ? "border-amber-300/30 bg-amber-300/[.07]" : "border-[#78e7f4]/25 bg-[#78e7f4]/[.06]"}`}
              >
                <Gauge
                  className={`mt-0.5 h-4 w-4 shrink-0 ${queueNotice.delayed ? "text-amber-200" : "text-[#78e7f4]"}`}
                />
                <div className="flex-1">
                  <p className="text-[11px] leading-5 text-slate-300">
                    <strong
                      className={
                        queueNotice.delayed
                          ? "text-amber-100"
                          : "text-[#b7fa59]"
                      }
                    >
                      {annotatedVideoUrl
                        ? "Verified replay ready."
                        : queueNotice.delayed
                          ? "Worker startup is taking longer than expected."
                          : "Analysis queued."}
                    </strong>{" "}
                    {annotatedVideoUrl
                      ? "The verified skeleton, shuttle, racket, and court-space overlays are ready for review."
                      : queueNotice.delayed
                        ? `RunPod has not picked up this job after about ${queueNotice.ageMinutes} minutes. Check the RunPod worker logs and Metrics tab; this is usually a worker initialization, capacity, image, or model-loading issue rather than a browser upload issue.`
                        : "The original video, selected player, court landmarks, and layer preferences were sent to the configured compute worker. This canvas shows verified skeleton, shuttle, racket, and court-space overlays only after the result returns."}
                  </p>
                  {sessionId && !annotatedVideoUrl && (
                    <button
                      onClick={refreshAnalysis}
                      disabled={refreshMutation.isPending}
                      className="mt-2 flex items-center gap-1.5 text-[10px] font-bold text-[#78e7f4] hover:text-[#b7fa59] disabled:opacity-50"
                    >
                      <RotateCcw
                        className={`h-3 w-3 ${refreshMutation.isPending ? "animate-spin" : ""}`}
                      />
                      {refreshMutation.isPending
                        ? "Checking worker..."
                        : "Check analysis status"}
                    </button>
                  )}
                </div>
              </div>
            )}
            {submissionState === "error" && (
              <div className="mt-4 flex items-start gap-3 rounded-xl border border-red-400/25 bg-red-400/[.06] p-3">
                <Info className="mt-0.5 h-4 w-4 shrink-0 text-red-300" />
                <p className="text-[11px] leading-5 text-slate-300">
                  <strong className="text-red-200">
                    Submission was not completed.
                  </strong>{" "}
                  The secure upload, session record, or worker could not accept
                  this analysis. Check the displayed error, then retry after
                  correcting the capture or worker connection.
                </p>
              </div>
            )}
          </section>

          <section className="panel-surface relative overflow-hidden rounded-2xl p-5">
            <div className="absolute -right-10 -top-10 h-40 w-40 rounded-full bg-[#b7fa59]/[.04] blur-2xl" />
            <div className="relative flex items-start justify-between">
              <div>
                <p className="font-mono-data text-[10px] uppercase tracking-[.16em] text-[#b7fa59]">
                  Output preview
                </p>
                <h2 className="mt-1 text-sm font-bold text-white">
                  Court-space trace
                </h2>
              </div>
              <Eye className="h-4 w-4 text-slate-500" />
            </div>
            <div className="relative mt-4 aspect-[1.66] rounded-xl border border-[#b7fa59]/15 bg-[#0b1619] p-3">
              <CourtBlueprint />
            </div>
            <div className="relative mt-4 grid grid-cols-3 gap-2">
              <div>
                <p className="font-mono-data text-[9px] uppercase text-slate-600">
                  Reference
                </p>
                <p className="mt-1 text-xs font-bold text-slate-300">
                  Court plane
                </p>
              </div>
              <div>
                <p className="font-mono-data text-[9px] uppercase text-slate-600">
                  Units
                </p>
                <p className="mt-1 text-xs font-bold text-slate-300">Metres</p>
              </div>
              <div>
                <p className="font-mono-data text-[9px] uppercase text-slate-600">
                  Camera
                </p>
                <p className="mt-1 text-xs font-bold text-slate-300">Pending</p>
              </div>
            </div>
          </section>
        </div>

        <section className="mt-5 grid gap-3 rounded-2xl border border-white/7 bg-white/[.018] p-4 md:grid-cols-3">
          {[
            {
              icon: Target,
              title: "Court-aware",
              text: "Movement is tied to real court geometry rather than raw image pixels.",
            },
            {
              icon: Clock3,
              title: "Frame accountable",
              text: "Every metric carries confidence and a link back to its source frames.",
            },
            {
              icon: Lock,
              title: "Private by design",
              text: "Session storage is isolated; exported material is controlled by the account owner.",
            },
          ].map(({ icon: Icon, title, text }) => (
            <div key={title} className="flex gap-3 px-2 py-1">
              <Icon className="mt-0.5 h-4 w-4 shrink-0 text-[#b7fa59]" />
              <p className="text-[11px] leading-5 text-slate-500">
                <strong className="font-bold text-slate-300">{title}. </strong>
                {text}
              </p>
            </div>
          ))}
        </section>

        {hasCompletedResult && (
          <>
            <section className="panel-surface mt-5 rounded-2xl p-5">
              <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-start">
                <div>
                  <p className="font-mono-data text-[10px] uppercase tracking-[.16em] text-[#b7fa59]">
                    Verified evidence report
                  </p>
                  <h2 className="mt-1 text-lg font-extrabold tracking-[-.035em] text-white">
                    Measurements, not a highlight reel.
                  </h2>
                </div>
                <div className="flex gap-2 font-mono-data text-[10px]">
                  <span className="rounded-md border border-white/10 px-2 py-1 text-slate-400">
                    Pose {Math.round((completedResult?.quality?.poseTrackConfidence ?? 0) * 100)}%
                  </span>
                  <span className="rounded-md border border-white/10 px-2 py-1 text-slate-400">
                    Shuttle {Math.round((completedResult?.quality?.shuttleTrackConfidence ?? 0) * 100)}%
                  </span>
                </div>
              </div>
              {(completedResult?.metrics?.length ?? 0) > 0 ? (
                <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                  {completedResult?.metrics?.map(item => (
                    <div key={item.metric} className="rounded-xl border border-white/8 bg-white/[.025] p-3">
                      <p className="text-[11px] font-bold text-slate-200">{item.metric}</p>
                      <p className="mt-1 font-mono-data text-lg text-[#d9ffad]">
                        {Number.isInteger(item.value) ? item.value : item.value.toFixed(2)} <span className="text-[10px] text-slate-500">{item.unit}</span>
                      </p>
                      <p className="mt-1 text-[10px] text-slate-500">Evidence confidence {Math.round(item.confidence * 100)}%</p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="mt-4 rounded-xl border border-amber-300/20 bg-amber-300/[.05] p-3 text-xs leading-5 text-amber-100/80">
                  No metric passed the evidence threshold for this footage. Review the capture guidance and the per-layer states before relying on coaching.
                </p>
              )}
              {(completedResult?.diagnostics?.warnings?.length ?? 0) > 0 && (
                <div className="mt-4 rounded-xl border border-white/8 bg-black/10 p-3">
                  <p className="font-mono-data text-[9px] uppercase tracking-[.14em] text-slate-500">Analysis notes</p>
                  <ul className="mt-2 space-y-1 text-[11px] leading-5 text-slate-400">
                    {completedResult?.diagnostics?.warnings?.map(note => <li key={note}>• {note}</li>)}
                  </ul>
                </div>
              )}
            </section>
          <section className="panel-surface mt-5 overflow-hidden rounded-2xl">
            <div className="grid lg:grid-cols-[.8fr_1.2fr]">
              <div className="border-b border-white/8 p-6 lg:border-b-0 lg:border-r">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#b7fa59]/10 text-[#b7fa59]">
                  <Sparkles className="h-5 w-5" />
                </div>
                <p className="mt-5 font-mono-data text-[10px] uppercase tracking-[.16em] text-[#b7fa59]">
                  DeepSeek coach
                </p>
                <h2 className="mt-2 text-2xl font-extrabold tracking-[-.045em] text-white">
                  Interrogate the evidence.
                </h2>
                <p className="mt-3 max-w-sm text-sm leading-6 text-slate-400">
                  Ask direct questions about the verified session. The coach is
                  instructed to cite evidence frames, expose uncertainty, and
                  avoid turning single-view data into false certainty.
                </p>
                <div className="mt-6 space-y-3">
                  <div className="flex items-center gap-3">
                    <span className="h-2 w-2 rounded-full bg-[#b7fa59]" />
                    <span className="text-xs text-slate-300">
                      Metric confidence is included in every answer
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="h-2 w-2 rounded-full bg-[#78e7f4]" />
                    <span className="text-xs text-slate-300">
                      Frame references anchor coaching claims
                    </span>
                  </div>
                </div>
              </div>
              <div className="min-h-[420px] p-4 sm:p-6">
                <AIChatBox
                  messages={coachMessages}
                  onSendMessage={askCoach}
                  isLoading={coachMutation.isPending}
                  height="370px"
                  className="border-white/10 bg-[#0b1419] shadow-none"
                  placeholder="Ask about timing, recovery, base position, or technique..."
                  emptyStateMessage="Your verified analysis is ready for a coaching question."
                  suggestedPrompts={[
                    "What is the highest-leverage technical change?",
                    "Where did my recovery break down?",
                    "Which evidence frames should I review first?",
                  ]}
                />
              </div>
            </div>
          </section>
          </>
        )}
      </main>
    </div>
  );
}
