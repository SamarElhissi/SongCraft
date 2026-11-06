import { PromptForm } from "./components/PromptForm";
import { SongList } from "./components/SongList";
import { useSongs } from "./hooks/useSongs";

export function App() {
  const songs = useSongs();

  return (
    <main>
      <header className="page-header">
        <h1>SongCraft</h1>
        <p>Submit a prompt. Generation keeps running if you close this page.</p>
      </header>
      <PromptForm />
      {songs.isPending && <p className="empty">Loading songs…</p>}
      {songs.isError && (
        <p className="error" role="alert">
          {songs.error.message}
        </p>
      )}
      {songs.data && <SongList songs={songs.data} />}
    </main>
  );
}
