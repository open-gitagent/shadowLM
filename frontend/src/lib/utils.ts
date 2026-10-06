import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

// Merge Tailwind classes with conflict resolution — the shadcn `cn` helper.
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
