import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
import { createSong } from "../api/client";
import type { Song } from "../api/types";

const MIN_PROMPT = 3;
const MAX_PROMPT = 500;

export function PromptForm() {
  const queryClient = useQueryClient();
  const [prompt, setPrompt] = useState("");
  const [duration, setDuration] = useState(30);
  const create = useMutation({
    mutationFn: createSong,
    onSuccess: (song) => {
      queryClient.setQueryData<Song[]>(["songs"], (current) => [song, ...(current ?? [])]);
      setPrompt("");
    },
  });
  const trimmed = prompt.trim();
  const promptReady = trimmed.length >= MIN_PROMPT && trimmed.length <= MAX_PROMPT;

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!promptReady || create.isPending) {
      return;
    }
    create.mutate({ prompt: trimmed, duration_seconds: duration });
  }

  return (
    <form className="panel" onSubmit={onSubmit}>
      <label htmlFor="prompt">Prompt</label>
      <textarea
        id="prompt"
        value={prompt}
        maxLength={MAX_PROMPT}
        rows={4}
        placeholder="A warm acoustic song about rain"
        onChange={(event) => setPrompt(event.target.value)}
      />
      <p className="hint">
        {trimmed.length}/{MAX_PROMPT}
        {trimmed.length > 0 && trimmed.length < MIN_PROMPT ? " · at least 3 characters" : ""}
      </p>
      <label htmlFor="duration">
        Duration <span className="duration-value">{duration}s</span>
      </label>
      <input
        id="duration"
        type="range"
        min={10}
        max={30}
        step={1}
        value={duration}
        onChange={(event) => setDuration(Number(event.target.value))}
      />
      <button type="submit" disabled={!promptReady || create.isPending}>
        {create.isPending ? "Submitting…" : "Generate"}
      </button>
      {create.isError && <p className="error">{create.error.message}</p>}
    </form>
  );
}
