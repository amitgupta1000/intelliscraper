import type { ScrapeResponse } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';

export async function startScrape(url: string): Promise<ScrapeResponse> {
  const res = await fetch(`${API_BASE_URL}/scrape`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Scrape failed');
  }
  return res.json();
}

export type { ScrapeResponse };
