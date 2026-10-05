import React, { type FormEvent, useState } from "react";
import { safeNextPath } from "@/lib/authUi";
import { trpc } from "@/lib/trpc";

type Mode = "sign_in" | "register";

const inputClass =
  "w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2.5 text-sm text-slate-100 outline-none placeholder:text-slate-500 focus:border-[#b7fa59]/60 focus:ring-2 focus:ring-[#b7fa59]/20";

export default function Login({ navigate = (path: string) => window.location.assign(path) }: { navigate?: (path: string) => void }) {
  const [mode, setMode] = useState<Mode>("sign_in");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const utils = trpc.useUtils();
  const login = trpc.auth.login.useMutation();
  const register = trpc.auth.register.useMutation();
  const busy = login.isPending || register.isPending;
  const registering = mode === "register";

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    try {
      if (registering) await register.mutateAsync({ name, email, password });
      else await login.mutateAsync({ email, password });
      await utils.auth.me.invalidate();
      navigate(safeNextPath(new URLSearchParams(window.location.search).get("next")));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Something went wrong. Try again.");
    }
  };

  return (
    <main className="flex min-h-screen items-center justify-center bg-[#0b1116] px-4 text-slate-100">
      <form onSubmit={submit} className="w-full max-w-sm rounded-2xl border border-white/10 bg-[#0d151b] p-6 shadow-2xl">
        <div className="mb-6 flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#b7fa59] font-bold text-[#0c1717]">O</div>
          <div>
            <h1 className="text-lg font-semibold">{registering ? "Create your account" : "Sign in to Optiqen"}</h1>
            <p className="text-xs text-slate-400">Your sessions and videos stay private to your account.</p>
          </div>
        </div>

        <div className="space-y-4">
          {registering && (
            <label className="block space-y-1.5 text-xs font-medium text-slate-300">
              <span>Name</span>
              <input className={inputClass} value={name} onChange={e => setName(e.target.value)} autoComplete="name" required maxLength={100} />
            </label>
          )}
          <label className="block space-y-1.5 text-xs font-medium text-slate-300">
            <span>Email</span>
            <input className={inputClass} type="email" value={email} onChange={e => setEmail(e.target.value)} autoComplete="email" required />
          </label>
          <label className="block space-y-1.5 text-xs font-medium text-slate-300">
            <span>Password</span>
            <input
              className={inputClass}
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              autoComplete={registering ? "new-password" : "current-password"}
              minLength={registering ? 8 : 1}
              required
            />
          </label>
        </div>

        {error && (
          <p role="alert" className="mt-4 rounded-lg border border-red-400/30 bg-red-500/10 px-3 py-2 text-xs text-red-200">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={busy}
          className="mt-6 w-full rounded-lg bg-[#b7fa59] py-2.5 text-sm font-semibold text-[#0c1717] transition-opacity hover:opacity-90 disabled:opacity-60"
        >
          {registering ? "Create account" : "Sign in"}
        </button>
        <button
          type="button"
          onClick={() => {
            setMode(registering ? "sign_in" : "register");
            setError(null);
          }}
          className="mt-3 w-full text-center text-xs text-slate-400 hover:text-slate-200"
        >
          {registering ? "I already have an account" : "Create an account"}
        </button>
      </form>
    </main>
  );
}
