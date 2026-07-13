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
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}/?XTransformPort=${BACKEND_PORT}`;
}

export function httpUrl(path: string) {
  return `${path}?XTransformPort=${BACKEND_PORT}`;
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
    try {
      this.ws = new WebSocket(wsUrl());
    } catch (e) {
      console.error('ws connect failed', e);
      this.scheduleReconnect();
      return;
    }
    this.ws.onopen = () => {
      this.connected = true;
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
        console.error('parse error', e);
      }
    };
    this.ws.onclose = () => {
      this.connected = false;
      this.scheduleReconnect();
    };
    this.ws.onerror = () => {
      try { this.ws?.close(); } catch {}
    };
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
