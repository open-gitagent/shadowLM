import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "@/app/App";
import { QueryClientProvider } from "@tanstack/react-query";
import { TooltipProvider } from "@/components/ui/tooltip";
import { queryClient } from "@/lib/queries";
import { embedded, pageToHash, startEmbed } from "@/lib/embed";

const render = () =>
  createRoot(document.getElementById("root")!).render(
    <StrictMode>
      <QueryClientProvider client={queryClient}>
        <TooltipProvider delayDuration={300}>
          <App />
        </TooltipProvider>
      </QueryClientProvider>
    </StrictMode>,
  );

// Embedded in a host console (lib/embed.ts), the studio waits for the host's
// first pass, opens the page the host asks for, and moves when told; the
// shell reports where it goes.
if (embedded) {
  const go = (path: string) => { window.location.hash = pageToHash(path); };
  void startEmbed(go).then((path) => {
    history.replaceState(null, "", `#${pageToHash(path)}`);
    render();
  });
} else {
  render();
}
