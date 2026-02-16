import { FormEvent, useState } from "react";

export interface ChatLine {
  id: string;
  role: "user" | "assistant";
  content: string;
}

export default function PromptConsole({
  lines,
  busy,
  onSend,
}: {
  lines: ChatLine[];
  busy: boolean;
  onSend: (prompt: string) => Promise<void>;
}): JSX.Element {
  const [prompt, setPrompt] = useState("");

  async function submit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    const value = prompt.trim();
    if (!value || busy) {
      return;
    }

    setPrompt("");
    await onSend(value);
  }

  return (
    <section className="panel prompt-panel reveal-slow">
      <header className="panel-header small">
        <p className="eyebrow">Prompt Link</p>
        <h3>Guidance Console</h3>
      </header>

      <div className="chat-log">
        {lines.length === 0 ? (
          <p className="empty-state">Issue commands like “switch to Europa capture plan”.</p>
        ) : (
          lines
            .slice()
            .reverse()
            .map((line) => (
              <article key={line.id} className={`chat-item ${line.role}`}>
                <p className="chat-role">{line.role === "user" ? "Operator" : "Loka"}</p>
                <p className="chat-content">{line.content}</p>
              </article>
            ))
        )}
      </div>

      <form className="prompt-form" onSubmit={submit}>
        <textarea
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          rows={3}
          placeholder="e.g. Re-plan insertion for Europa and prioritize propellant margin under 25%."
        />
        <button type="submit" className="btn primary" disabled={busy}>
          {busy ? "Sending..." : "Send Prompt"}
        </button>
      </form>
    </section>
  );
}
