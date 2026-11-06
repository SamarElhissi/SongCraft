import type { Song } from "../api/types";
import { SongCard } from "./SongCard";

export function SongList({ songs }: { songs: Song[] }) {
  if (songs.length === 0) {
    return <p className="empty">No songs yet.</p>;
  }
  return (
    <section className="song-list">
      {songs.map((song) => (
        <SongCard key={song.id} song={song} />
      ))}
    </section>
  );
}
