'use client';

/**
 * NEAT visualizer client. Connects to the Python backend via WebSocket
 * (routed through Caddy with XTransformPort=8000).
 */

export type EpisodeTrace = {
  obs: number[][];
  actions: number[];
  reward: number;
  steps: number;
};

export type GenomeNode = {
  kind: string;
  activation: string;
};

export type GenomeConn = {
  in: number;
  out: number;
  w: number;
  en: boolean;
};

export type GenomeDict = {
  n_inputs: number;
  n_outputs: number;
  nodes: Record<string, GenomeNode>;
  conns: Record<string, { in: number; out: number; w: number; en: boolean }>;
  fitness: number;
  species_id: number;
  generation: number;
};

export type SpeciesInfo = {
  id: number;
  size: number;
  best_fitness: number;
  staleness: number;
  avg_fitness: number;
  representative: GenomeDict | null;
};

export type HistoryEntry = {
  generation: number;
  best_fitness: number;
  mean_fitness: number;
  n_species: number;
  population_size: number;
  avg_conns: number;
  avg_nodes: number;
  species_threshold: number;
};

export type Snapshot = {
  generation: number;
  best_fitness: number;
  history: HistoryEntry[];
  n_species: number;
  threshold: number;
  species: SpeciesInfo[];
  best_genome: GenomeDict | null;
  genomes: GenomeDict[];
  running?: boolean;
  episode_buffer?: Array<{ genome_id: number; seed: number; trace: EpisodeTrace; generation: number }>;
};

const BACKEND_PORT = 8000;

function wsUrl() {
  if (typeof window === 'undefined') return '';
  // Use relative path - Next.js rewrites will proxy /ws to the backend.
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}/ws`;
}

export function httpUrl(path: string) {
  // Use relative path - Next.js rewrites will proxy /api/* to the backend.
  return path;
}

export class NeatClient {
  ws: WebSocket | null = null;
  listeners: Set<(snap: Snapshot) => void> = new Set();
  statusListeners: Set<(running: boolean) => void> = new Set();
  connected: boolean = false;
  reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  connect() {
    if (typeof window === 'undefined') return;
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) return;
    const url = wsUrl();
    console.log('[NeatClient] connecting to', url);
    try {
      this.ws = new WebSocket(url);
    } catch (e) {
      console.error('[NeatClient] ws connect failed', e);
      this.startPolling();
      return;
    }
    let wsConnected = false;
    this.ws.onopen = () => {
      this.connected = true;
      wsConnected = true;
      console.log('[NeatClient] connected');
      // stop polling if it was running
      this.stopPolling();
    };
    this.ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data.type === 'heartbeat') {
          this.statusListeners.forEach((l) => l(!!data.running));
          return;
        }
        this.listeners.forEach((l) => l(data as Snapshot));
        if (typeof data.running === 'boolean') {
          this.statusListeners.forEach((l) => l(data.running));
        }
      } catch (e) {
        console.error('[NeatClient] parse error', e);
      }
    };
    this.ws.onclose = (ev) => {
      this.connected = false;
      console.log('[NeatClient] closed', ev.code, ev.reason);
      // fall back to polling if WS doesn't work
      if (!wsConnected) {
        console.log('[NeatClient] falling back to polling');
        this.startPolling();
      } else {
        this.scheduleReconnect();
      }
    };
    this.ws.onerror = (e) => {
      console.error('[NeatClient] ws error', e);
      try { this.ws?.close(); } catch {}
    };
    // safety: if WS doesn't connect within 3s, start polling
    setTimeout(() => {
      if (!wsConnected && (!this.ws || this.ws.readyState !== WebSocket.OPEN)) {
        console.log('[NeatClient] WS timeout, starting polling fallback');
        this.startPolling();
      }
    }, 3000);
  }

  pollingTimer: ReturnType<typeof setInterval> | null = null;

  startPolling() {
    if (this.pollingTimer) return;
    this.pollingTimer = setInterval(async () => {
      try {
        const res = await fetch(httpUrl('/api/state'));
        if (res.ok) {
          const data = await res.json();
          this.listeners.forEach((l) => l(data as Snapshot));
          if (typeof data.running === 'boolean') {
            this.statusListeners.forEach((l) => l(data.running));
          }
        }
      } catch (e) {
        // ignore
      }
    }, 1000);
  }

  stopPolling() {
    if (this.pollingTimer) {
      clearInterval(this.pollingTimer);
      this.pollingTimer = null;
    }
  }

  scheduleReconnect() {
    if (this.reconnectTimer) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, 2000);
  }

  disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.stopPolling();
    if (this.ws) {
      try { this.ws.close(); } catch {}
      this.ws = null;
    }
  }

  onSnapshot(cb: (snap: Snapshot) => void) {
    this.listeners.add(cb);
    return () => this.listeners.delete(cb);
  }

  onStatus(cb: (running: boolean) => void) {
    this.statusListeners.add(cb);
    return () => this.statusListeners.delete(cb);
  }

  async start(config: Record<string, unknown>) {
    const res = await fetch(httpUrl('/api/start'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    return res.json();
  }

  async stop() {
    const res = await fetch(httpUrl('/api/stop'), { method: 'POST' });
    return res.json();
  }

  async setDelay(delay: number) {
    const res = await fetch(httpUrl('/api/set_delay'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ delay }),
    });
    return res.json();
  }

  async getState(): Promise<Snapshot | null> {
    try {
      const res = await fetch(httpUrl('/api/state'));
      if (!res.ok) return null;
      return await res.json();
    } catch {
      return null;
    }
  }
}

// Singleton
let _client: NeatClient | null = null;
export function getNeatClient() {
  if (!_client) _client = new NeatClient();
  return _client;
}

// ----------------------------------------------------------------- Playback -
/**
 * Playback client: loads pre-recorded training data from /training-data.json
 * and plays through snapshots at a configurable speed. Used when the live
 * Python backend cannot run (e.g. in restricted sandbox environments).
 */
export class PlaybackClient {
  snapshots: Snapshot[] = [];
  index: number = 0;
  listeners: Set<(snap: Snapshot) => void> = new Set();
  statusListeners: Set<(running: boolean) => void> = new Set();
  playing: boolean = false;
  speed: number = 1; // generations per second
  timer: ReturnType<typeof setInterval> | null = null;
  loaded: boolean = false;
  config: Record<string, unknown> = {};

  async load() {
    if (this.loaded) return;
    try {
      const res = await fetch('/training-data.json');
      const data = await res.json();
      this.snapshots = data.snapshots || [];
      this.config = data.config || {};
      this.loaded = true;
      console.log(`[PlaybackClient] loaded ${this.snapshots.length} snapshots`);
    } catch (e) {
      console.error('[PlaybackClient] failed to load training data', e);
    }
  }

  onSnapshot(cb: (snap: Snapshot) => void) {
    this.listeners.add(cb);
    // send current snapshot immediately if loaded
    if (this.loaded && this.snapshots.length > 0) {
      cb(this.snapshots[this.index]);
    }
    return () => this.listeners.delete(cb);
  }

  onStatus(cb: (running: boolean) => void) {
    this.statusListeners.add(cb);
    return () => this.statusListeners.delete(cb);
  }

  play() {
    if (!this.loaded || this.snapshots.length === 0) return;
    this.playing = true;
    this.statusListeners.forEach((l) => l(true));
    const intervalMs = 1000 / Math.max(0.1, this.speed);
    if (this.timer) clearInterval(this.timer);
    this.timer = setInterval(() => {
      if (this.index >= this.snapshots.length - 1) {
        this.pause();
        return;
      }
      this.index++;
      const snap = this.snapshots[this.index];
      this.listeners.forEach((l) => l(snap));
    }, intervalMs);
  }

  pause() {
    this.playing = false;
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    this.statusListeners.forEach((l) => l(false));
  }

  setSpeed(speed: number) {
    this.speed = speed;
    if (this.playing) {
      this.play(); // restart with new speed
    }
  }

  seek(index: number) {
    this.index = Math.max(0, Math.min(index, this.snapshots.length - 1));
    if (this.snapshots[this.index]) {
      this.listeners.forEach((l) => l(this.snapshots[this.index]));
    }
  }

  async start(_config: Record<string, unknown>) {
    await this.load();
    this.index = 0;
    if (this.snapshots.length > 0) {
      this.listeners.forEach((l) => l(this.snapshots[0]));
    }
    this.play();
    return { status: 'started' };
  }

  async stop() {
    this.pause();
    return { status: 'stopped' };
  }

  async setDelay(delay: number) {
    // delay in seconds = 1/speed
    if (delay <= 0) {
      this.setSpeed(1000); // very fast
    } else {
      this.setSpeed(1 / delay);
    }
    return { delay };
  }

  async getState(): Promise<Snapshot | null> {
    if (!this.loaded || this.snapshots.length === 0) return null;
    return this.snapshots[this.index];
  }
}

let _playbackClient: PlaybackClient | null = null;
export function getPlaybackClient() {
  if (!_playbackClient) _playbackClient = new PlaybackClient();
  return _playbackClient;
}
