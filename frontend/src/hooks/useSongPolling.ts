import { useQuery, useQueryClient } from "@tanstack/react-query";
import { getSong } from "../api/client";
import { type Song, type SongStatus } from "../api/types";

export function refetchIntervalMs(status: SongStatus | undefined): number | false {
  if (status === "completed" || status === "failed") {
    return false;
  }
  return 2000;
}

export function useSongPolling(songId: string, enabled: boolean) {
  const queryClient = useQueryClient();
  return useQuery({
    queryKey: ["song", songId],
    enabled,
    refetchIntervalInBackground: true,
    queryFn: async () => {
      const song = await getSong(songId);
      queryClient.setQueryData<Song[]>(["songs"], (current) =>
        current?.map((item) => (item.id === song.id ? song : item)),
      );
      return song;
    },
    refetchInterval: (query) => refetchIntervalMs(query.state.data?.status),
  });
}
