"use client";

import type React from "react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Send } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

// ===== Types =====
export type MsgStatus = "sent" | "delivered" | "read";
export type Role = "user" | "ai";

export interface Message {
  id: string;
  role: Role;
  text: string;
  at: string;
  status?: MsgStatus;
}

// ===== Utils =====
function timeLabel(iso: string) {
  const d = new Date(iso);
  return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

// ===== UI parts =====
function AIBubble({ message }: { message: Message }) {
  return (
    <div className="flex w-full justify-start">
      <div className="max-w-[80%]">
        <div className="rounded-2xl bg-muted px-3 py-2 text-sm shadow-sm">
          <p className="whitespace-pre-wrap break-words leading-relaxed">{message.text}</p>
        </div>
        <div className="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
          <span>{timeLabel(message.at)}</span>
        </div>
      </div>
    </div>
  );
}

function UserBubble({ message }: { message: Message }) {
  return (
    <div className="flex w-full justify-end">
      <div className="max-w-[80%]">
        <div className="rounded-2xl bg-primary px-3 py-2 text-sm text-primary-foreground shadow-sm">
          <p className="whitespace-pre-wrap break-words leading-relaxed">{message.text}</p>
        </div>
        <div className="mt-1 flex items-center justify-end gap-1 text-xs text-muted-foreground">
          <span>{timeLabel(message.at)}</span>
        </div>
      </div>
    </div>
  );
}

function MessageList({ messages, scrollRef }: { messages: Message[]; scrollRef: React.RefObject<HTMLDivElement> }) {
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages?.length]);

  return (
    <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-3">
      <div className="flex flex-col gap-4">
        {messages.map((m) => (m.role === "ai" ? <AIBubble key={m.id} message={m} /> : <UserBubble key={m.id} message={m} />))}
        <div className="h-1" />
      </div>
    </div>
  );
}

function Composer({ value, onChange, onSend, placeholder }: { value: string; onChange: (v: string) => void; onSend: () => void; placeholder: string }) {
  return (
    <div className="border-t p-3">
      <div className="flex items-end gap-2">
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
    </div>
  );
}

// ===== Main =====
export default function ChatApp() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");

  const scrollRef = useRef<HTMLDivElement>(null);

  const appendMessage = useCallback((msg: Message) => {
    setMessages((prev) => [...prev, msg]);
  }, []);

  const onSend = useCallback(async () => {
    const text = draft.trim();
    if (!text) return;
    const userMsg: Message = { id: crypto.randomUUID(), role: "user", text, at: new Date().toISOString(), status: "delivered" };
    appendMessage(userMsg);
    setDraft("");

    try {
      const res = await fetch("http://localhost:8000/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text })
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const replyText = data.hits?.[0]?.text ?? "検索結果が見つかりませんでした。";
      const aiMsg: Message = {
        id: crypto.randomUUID(),
        role: "ai",
        text: replyText,
        at: new Date().toISOString(),
        status: "read",
      };
      setMessages((prev) => [...prev, aiMsg]);
    } catch (err: any) {
      const aiMsg: Message = {
        id: crypto.randomUUID(),
        role: "ai",
        text: `エラー: ${err.message}`,
        at: new Date().toISOString(),
        status: "read",
      };
      setMessages((prev) => [...prev, aiMsg]);
    }
  }, [appendMessage, draft]);

  return (
    <div className="flex h-[720px] w-full flex-col overflow-hidden rounded-2xl border bg-background shadow-sm">
      <MessageList messages={messages} scrollRef={scrollRef} />
      <Composer value={draft} onChange={setDraft} onSend={onSend} placeholder={"ドキュメントに基づいて質問…"} />
    </div>
  );
}
