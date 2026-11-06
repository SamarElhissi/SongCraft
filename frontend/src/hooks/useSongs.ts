import { useQuery } from "@tanstack/react-query";
import { listSongs } from "../api/client";

export function useSongs() {
  return useQuery({
    queryKey: ["songs"],
    queryFn: listSongs,
  });
}
