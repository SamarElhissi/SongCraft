import { useMutation, useQueryClient } from "@tanstack/react-query";
import { audioUrl, deleteSong } from "../api/client";
import { isFinished, type Song } from "../api/types";
import { useSongPolling } from "../hooks/useSongPolling";
import { ProgressSteps } from "./ProgressSteps";

export function SongCard({ song }: { song: Song }) {
  useSongPolling(song.id, !isFinished(song.status));
  const queryClient = useQueryClient();
  const remove = useMutation({
    mutationFn: () => deleteSong(song.id),
    onSuccess: () => {
      queryClient.setQueryData<Song[]>(["songs"], (current) =>
        current?.filter((item) => item.id !== song.id),
      );
      queryClient.removeQueries({ queryKey: ["song", song.id] });
    },
  });

  function onDelete() {
    if (window.confirm("Delete this song?")) {
      remove.mutate();
    }
  }

  return (
    <article className={`song song-${song.status}`}>
      <header className="song-header">
        <h2>{song.prompt}</h2>
        <button type="button" onClick={onDelete} disabled={remove.isPending}>
          {remove.isPending ? "Deleting…" : "Delete"}
        </button>
      </header>
      <p className="meta">
        {song.duration_seconds}s
        {song.tempo_bpm !== null && song.status === "completed" ? ` · ${Math.round(song.tempo_bpm)} BPM` : ""}
      </p>
      {!isFinished(song.status) && <ProgressSteps song={song} />}
      {song.status === "failed" && (
        <p className="error" role="alert">
          {song.error_message ?? "Song generation failed"}
        </p>
      )}
      {song.lyrics && <pre className="lyrics">{song.lyrics}</pre>}
      <div className="players">
        {song.tracks.final && (
          <label>
            Final mix
            <audio controls preload="none" src={audioUrl(song.id, "final")} />
          </label>
        )}
        {song.tracks.instrumental && (
          <label>
            Instrumental
            <audio controls preload="none" src={audioUrl(song.id, "instrumental")} />
          </label>
        )}
      </div>
      {remove.isError && <p className="error">{remove.error.message}</p>}
    </article>
  );
}
