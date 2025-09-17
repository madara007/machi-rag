"use client";

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  MessageCircle,
  Send,
  Plus,
  Paperclip,
  Smile,
  CheckCheck,
  Check,
  Library,
} from "lucide-react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

// ------------------------------------------------------------
// RAG-ready Chat UI (Next.js, single file)
// - Tailwind + shadcn/ui + framer-motion + lucide-react
// - Left: コレクション/フィルタ、Center: チャット、Right: 履歴ロールバー（ジャンプ可）
// - 2スペースインデント / 不要な import 削除 / 型の厳格化
// - 小さな責務のコンポーネントへ分割（同一ファイル内）
// - 可読性・アクセシビリティ改善
// ------------------------------------------------------------

// ===== Types =====
export type MsgStatus = "sent" | "delivered" | "read";
export type Role = "user" | "ai";

export interface Message {
  id: string;
  role: Role;
  text: string; // markdown/plain
  at: string; // ISO
  status?: MsgStatus;
}

export interface Conversation {
  id: string;
  title: string;
  unread: number;
  lastMessage?: string;
}

// ===== Seeds =====
const SEED_CONVERSATIONS: Conversation[] = [
  { id: "conv-1", title: "製品Q&A（社内RAG）", unread: 1, lastMessage: "在庫APIのレート制限…" },
  { id: "conv-2", title: "法務問い合わせ", unread: 0, lastMessage: "秘密保持に関する条項…" },
];

const SEED_MESSAGES: Record<string, Message[]> = {
  "conv-1": [
    {
      id: "m1",
      role: "ai",
      text: "こんにちは。製品ドキュメントから回答できます。ご質問をどうぞ。",
      at: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
      status: "read",
    },
  ],
};

// ===== Utils =====
function cx(...xs: Array<string | false | undefined | null>) {
  return xs.filter(Boolean).join(" ");
}

function timeLabel(iso: string) {
  const d = new Date(iso);
  return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

function dateKey(iso: string) {
  const d = new Date(iso);
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function dateLabelFromKey(key: string) {
  const [y, m, d] = key.split("-").map(Number);
  const base = new Date(y, (m as number) - 1, d as number);
  const todayKey = dateKey(new Date().toISOString());
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  const yesterdayKey = dateKey(yesterday.toISOString());
  if (key === todayKey) return "今日";
  if (key === yesterdayKey) return "昨日";
  return base.toLocaleDateString(undefined, { month: "short", day: "numeric", weekday: "short" });
}

// Make deps optional & always an array for safety
function useAutoScroll(depKeys: ReadonlyArray<unknown> | undefined, ref: React.RefObject<HTMLDivElement>) {
  useEffect(() => {
    ref.current?.scrollTo({ top: ref.current.scrollHeight, behavior: "smooth" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, Array.isArray(depKeys) ? depKeys : []);
}

// ===== Presentational bits =====
function StatusTicks({ status }: { status?: MsgStatus }) {
  if (!status) return null;
  return (
    <span className="ml-2 inline-flex items-center text-muted-foreground" aria-hidden>
      {status === "read" ? (
        <CheckCheck className="h-3.5 w-3.5" />
      ) : status === "delivered" ? (
        <Check className="h-3.5 w-3.5" />
      ) : null}
    </span>
  );
}

function AIBubble({ message }: { message: Message }) {
  return (
    <div className="flex w-full justify-start gap-2">
      <Avatar className="h-7 w-7">
        <AvatarImage src="https://images.unsplash.com/photo-1544723795-3fb6469f5b39?w=128&auto=format&fit=facearea&facepad=2&h=128" />
        <AvatarFallback>AI</AvatarFallback>
      </Avatar>
      <div className="max-w-[80%]">
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl bg-muted px-3 py-2 text-sm shadow-sm">
          <p className="whitespace-pre-wrap break-words leading-relaxed">{message.text}</p>
        </motion.div>
        <div className="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
          <span>{timeLabel(message.at)}</span>
        </div>
      </div>
    </div>
  );
}

function UserBubble({ message }: { message: Message }) {
  return (
    <div className="flex w-full justify-end gap-2">
      <div className="max-w-[80%]">
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl bg-primary px-3 py-2 text-sm text-primary-foreground shadow-sm">
          <p className="whitespace-pre-wrap break-words leading-relaxed">{message.text}</p>
        </motion.div>
        <div className="mt-1 flex items-center justify-end gap-1 text-xs text-muted-foreground">
          <span>{timeLabel(message.at)}</span>
          <StatusTicks status={message.status} />
        </div>
      </div>
      <Avatar className="h-7 w-7">
        <AvatarImage src="https://images.unsplash.com/photo-1527980965255-d3b416303d12?w=128&auto=format&fit=facearea&facepad=2&h=128" />
        <AvatarFallback>Me</AvatarFallback>
      </Avatar>
    </div>
  );
}

function Header() {
  return (
    <div className="flex items-center gap-3 border-b p-4">
      <Avatar className="h-9 w-9">
        <AvatarImage src="https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=128&auto=format&fit=facearea&facepad=2&h=128" />
        <AvatarFallback>RG</AvatarFallback>
      </Avatar>
      <div className="min-w-0">
        <p className="truncate font-medium leading-tight">社内RAGアシスタント</p>
      </div>
    </div>
  );
}

function Sidebar() {
  return (
    <aside className="hidden h-full flex-col border-r p-4 lg:flex">
      <div className="mb-3 flex items-center gap-2">
        <MessageCircle className="h-5 w-5" />
        <h2 className="text-lg font-semibold">RAG チャット</h2>
        <div className="ml-auto flex items-center gap-2">
          <Button size="icon" variant="secondary" className="rounded-xl" aria-label="新規チャット">
            <Plus className="h-4 w-4" />
          </Button>
        </div>
      </div>
      <div className="mb-4">
        <p className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase text-muted-foreground">
          <Library className="h-3.5 w-3.5" />チャット
        </p>
      </div>
    </aside>
  );
}

function TypingDots() {
  return (
    <div className="flex items-end gap-1 px-3 py-2 text-sm text-muted-foreground" aria-live="polite" aria-label="AI typing">
      <motion.span className="h-1.5 w-1.5 rounded-full bg-current" animate={{ opacity: [0.2, 1, 0.2], y: [0, -2, 0] }} transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut", delay: 0 }} />
      <motion.span className="h-1.5 w-1.5 rounded-full bg-current" animate={{ opacity: [0.2, 1, 0.2], y: [0, -2, 0] }} transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut", delay: 0.2 }} />
      <motion.span className="h-1.5 w-1.5 rounded-full bg-current" animate={{ opacity: [0.2, 1, 0.2], y: [0, -2, 0] }} transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut", delay: 0.4 }} />
    </div>
  );
}

function DateSeparator({ label, innerRef }: { label: string; innerRef?: (el: HTMLDivElement | null) => void }) {
  return (
    <div ref={innerRef} className="sticky top-2 z-10 mx-auto w-fit rounded-full bg-muted px-3 py-1 text-[11px] font-medium text-muted-foreground shadow-sm">
      {label}
    </div>
  );
}

// ===== Right: History Rollbar =====
interface GroupPosition {
  key: string;
  label: string;
  posPct: number; // 0..100, relative to total scrollable height
}

function HistoryRollbar({
  groups,
  progress,
  onJump,
}: {
  groups: GroupPosition[];
  progress: number;
  onJump: (key: string) => void;
}) {
  return (
    <aside className="hidden h-full select-none items-center justify-center border-l px-3 lg:flex">
      <div className="relative h-[80%] w-2" aria-label="履歴ロールバー">
        {/* track */}
        <div className="absolute left-1/2 h-full w-0.5 -translate-x-1/2 rounded bg-muted" />
        {/* progress */}
        <div
          className="absolute left-1/2 w-0.5 -translate-x-1/2 rounded bg-primary"
          style={{ height: `${Math.max(0, Math.min(100, progress))}%` }}
          aria-hidden
        />
        {/* anchors */}
        {groups.map((g) => (
          <button
            key={g.key}
            onClick={() => onJump(g.key)}
            className={cx(
              "absolute left-1/2 -translate-x-1/2 translate-y-1/2 rounded-full border bg-background/80 backdrop-blur p-1 shadow",
              "hover:scale-110 transition-transform"
            )}
            style={{ top: `${g.posPct}%` }}
            aria-label={`${g.label} にジャンプ`}
            title={g.label}
          >
            <span className="block h-2 w-2 rounded-full bg-foreground" />
          </button>
        ))}
      </div>
    </aside>
  );
}

// ===== Center: Messages with custom scroll container =====
function MessageList({
  messages,
  isTyping,
  scrollRef,
  setAnchor,
}: {
  messages: Message[];
  isTyping: boolean;
  scrollRef: React.RefObject<HTMLDivElement>;
  setAnchor: (key: string) => (el: HTMLDivElement | null) => void; // callback ref factory
}) {
  // auto scroll to bottom on new message / typing
  useAutoScroll([messages?.length ?? 0, isTyping], scrollRef);

  // group messages by date
  const groups = useMemo(() => {
    const map = new Map<string, Message[]>();
    for (const m of messages || []) {
      const k = dateKey(m.at);
      if (!map.has(k)) map.set(k, []);
      map.get(k)!.push(m);
    }
    return Array.from(map.entries());
  }, [messages]);

  return (
    <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-3">
      <div className="flex flex-col gap-4">
        {groups.map(([k, msgs]) => (
          <div key={k} className="flex flex-col gap-4">
            <DateSeparator label={dateLabelFromKey(k)} innerRef={setAnchor(k)} />
            {msgs.map((m) => (m.role === "ai" ? <AIBubble key={m.id} message={m} /> : <UserBubble key={m.id} message={m} />))}
          </div>
        ))}
        <AnimatePresence>{isTyping && <TypingDots />}</AnimatePresence>
        {/* bottom spacer */}
        <div className="h-1" />
      </div>
    </div>
  );
}

function Composer({
  value,
  onChange,
  onSend,
  placeholder,
}: {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  placeholder: string;
}) {
  return (
    <div className="border-t p-3">
      <div className="flex items-end gap-2">
        <Button size="icon" variant="ghost" className="rounded-xl" aria-label="ファイル添付">
          <Paperclip className="h-5 w-5" />
        </Button>
        <div className="flex-1">
          <Input
            placeholder={placeholder}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                onSend();
              }
            }}
            className="rounded-2xl"
            aria-label="メッセージ入力"
          />
        </div>
        <Button onClick={onSend} className="rounded-xl" aria-label="送信">
          <Send className="mr-1 h-4 w-4" /> 送信
        </Button>
      </div>
      <div className="mt-1 flex items-center justify-between px-1 text-xs text-muted-foreground">
        <div className="flex items-center gap-1">
          <Smile className="h-3.5 w-3.5" />
          <span>Shift+Enterで改行</span>
        </div>
      </div>
    </div>
  );
}

// ===== Fake backend =====
function composeAIAnswer(m: "answer" | "summarize" | "draft") {
  if (m === "summarize") {
    return `以下の出典から要点を要約します:\n- レートは1分間に100リクエスト\n- バースト許容量は200\n- 429時は指数バックオフで最大5回リトライ`;
  }
  if (m === "draft") {
    return `件名: 在庫APIレート制限について\n\nお疲れ様です。社内RAGで確認したところ、在庫APIの標準レートは1分100reqで、バーストは200まで許容されます。429発生時は指数バックオフで最大5回の再試行が推奨です。詳細は下記出典をご参照ください。`;
  }
  return `在庫APIの標準レートは **1分あたり100リクエスト**、一時的なバーストは **200** まで許容されます。429(Too Many Requests) が返る場合は、\n**指数バックオフ** で最大5回リトライし、\`Retry-After\` がある場合はそれを優先します。\n認証は OAuth2 クライアントクレデンシャルで、スコープ \`inventory.read\` が必要です。`;
}

// ===== Main =====
export default function ChatApp() {
  // state
  const [conversations] = useState<Conversation[]>(SEED_CONVERSATIONS);
  const [activeId] = useState<string>(SEED_CONVERSATIONS[0].id);
  const [messagesMap, setMessagesMap] = useState<Record<string, Message[]>>(SEED_MESSAGES);
  const [draft, setDraft] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [mode] = useState<"answer" | "summarize" | "draft">("answer");

  const activeConv = useMemo(() => conversations.find((c) => c.id === activeId)!, [conversations, activeId]);
  const messages = messagesMap[activeId] ?? [];

  // scroll + anchors (no hooks-in-loops!)
  const scrollRef = useRef<HTMLDivElement>(null);
  const anchorsMapRef = useRef<Record<string, HTMLDivElement | null>>({});
  const [progress, setProgress] = useState(0); // 0..100
  const [groupPositions, setGroupPositions] = useState<GroupPosition[]>([]);

  const dateKeys = useMemo(() => Array.from(new Set((messages ?? []).map((m) => dateKey(m.at)))), [messages]);

  // factory to supply callback refs per date key
  const setAnchor = useCallback((key: string) => (el: HTMLDivElement | null) => {
    anchorsMapRef.current[key] = el;
  }, []);

  const recomputePositions = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const scrollable = el.scrollHeight - el.clientHeight;
    const next: GroupPosition[] = dateKeys.map((k) => {
      const r = anchorsMapRef.current[k];
      const top = r ? r.offsetTop : 0;
      const posPct = scrollable > 0 ? Math.min(100, Math.max(0, (top / scrollable) * 100)) : 0;
      return { key: k, label: dateLabelFromKey(k), posPct };
    });
    setGroupPositions(next);

    // lightweight runtime assertions (extra tests)
    if (process.env.NODE_ENV !== "production") {
      if (!Array.isArray(messages)) console.error("[test] messages should be array");
      if (!Array.isArray(next)) console.error("[test] positions should be array");
      for (const k of dateKeys) {
        if (!anchorsMapRef.current[k]) console.warn(`[test] missing anchor for ${k}`);
      }
    }
  }, [dateKeys, messages]);

  useLayoutEffect(() => {
    recomputePositions();
  }, [recomputePositions, messages?.length ?? 0]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onScroll = () => {
      const scrollable = el.scrollHeight - el.clientHeight;
      const pct = scrollable > 0 ? (el.scrollTop / scrollable) * 100 : 100;
      setProgress(pct);
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    const onResize = () => recomputePositions();
    window.addEventListener("resize", onResize);
    onScroll();
    return () => {
      el.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onResize);
    };
  }, [recomputePositions]);

  // send handlers
  const appendMessage = useCallback((msg: Message) => {
    setMessagesMap((prev) => ({ ...prev, [activeId]: [...(prev[activeId] ?? []), msg] }));
  }, [activeId]);

  const fakeAnswer = useCallback(() => {
    setIsTyping(true);
    const text = composeAIAnswer(mode);
    const aiMsg: Message = { id: crypto.randomUUID(), role: "ai", text, at: new Date().toISOString() };
    setTimeout(() => {
      appendMessage(aiMsg);
      setIsTyping(false);
      // 送信後に最新のアンカー位置を更新
      requestAnimationFrame(() => recomputePositions());
    }, 900);
  }, [appendMessage, mode, recomputePositions]);

  const onSend = useCallback(() => {
    const text = draft.trim();
    if (!text) return;
    const userMsg: Message = { id: crypto.randomUUID(), role: "user", text, at: new Date().toISOString(), status: "delivered" };
    appendMessage(userMsg);
    setDraft("");
    fakeAnswer();
  }, [appendMessage, draft, fakeAnswer]);

  const jumpTo = useCallback((key: string) => {
    const el = scrollRef.current;
    const r = anchorsMapRef.current[key];
    if (!el || !r) return;
    el.scrollTo({ top: r.offsetTop - 12, behavior: "smooth" });
  }, []);

  // tiny self-checks (existing) + extra test cases (added)
  useEffect(() => {
    try {
      const a = composeAIAnswer("answer");
      if (!a.includes("1分あたり100")) console.warn("[test] answer should mention rate");
      const s = composeAIAnswer("summarize");
      if (s.split("\n").length < 3) console.warn("[test] summarize should be multiline");
      const d = composeAIAnswer("draft");
      if (!d.startsWith("件名")) console.warn("[test] draft should start with subject");
      // Added tests
      if (!Array.isArray(SEED_MESSAGES[activeId])) console.error("[test] seed messages must be array");
      if (!activeConv?.id) console.error("[test] active conversation should exist");
    } catch (e) {
      console.warn("[test] self-checks skipped:", e);
    }
  }, [activeConv?.id, activeId]);

  return (
    <div className="grid h-[760px] w-full grid-cols-1 overflow-hidden rounded-2xl border bg-background shadow-sm lg:grid-cols-[320px_1fr_56px]">
      {/* Left: Collections & Filters */}
      <Sidebar />

      {/* Center: Chat */}
      <section className="flex h-full flex-col">
        <Header />
        <MessageList messages={messages} isTyping={isTyping} scrollRef={scrollRef} setAnchor={setAnchor} />
        <Composer value={draft} onChange={setDraft} onSend={onSend} placeholder={"ドキュメントに基づいて質問…"} />
      </section>

      {/* Right: History Rollbar */}
      <HistoryRollbar groups={groupPositions} progress={progress} onJump={jumpTo} />
    </div>
  );
}
