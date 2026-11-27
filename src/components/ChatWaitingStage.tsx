import {
  useEffect,
  useRef,
  type MouseEvent,
  type PointerEvent,
} from 'react';
import { BookOpen } from 'lucide-react';

interface ChatWaitingStageProps {
  chatTitle: string;
  hasDraft: boolean;
  onFocusComposer: () => void;
}

interface Mote {
  angle: number;
  baseRadiusX: number;
  baseRadiusY: number;
  speed: number;
  size: number;
  kind: 'dot' | 'diamond' | 'plus';
  tone: 'sky' | 'indigo' | 'slate';
  x: number;
  y: number;
  vx: number;
  vy: number;
  alpha: number;
}

interface RippleWave {
  x: number;
  y: number;
  radius: number;
  maxRadius: number;
  alpha: number;
  speed: number;
  tone: 'sky' | 'indigo';
}

export function ChatWaitingStage({
  chatTitle,
  hasDraft,
  onFocusComposer,
}: ChatWaitingStageProps) {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const beaconRef = useRef<HTMLButtonElement | null>(null);

  const hasDraftRef = useRef(hasDraft);
  hasDraftRef.current = hasDraft;

  const pointerRef = useRef<{ x: number; y: number; active: boolean }>({
    x: 0,
    y: 0,
    active: false,
  });

  const ripplesRef = useRef<RippleWave[]>([]);

  const spawnRipple = (
    x: number,
    y: number,
    options?: { double?: boolean; tone?: 'sky' | 'indigo' }
  ) => {
    ripplesRef.current.push({
      x,
      y,
      radius: 8,
      maxRadius: 175,
      alpha: 0.42,
      speed: 2.9,
      tone: options?.tone ?? 'sky',
    });
    if (options?.double) {
      ripplesRef.current.push({
        x,
        y,
        radius: 2,
        maxRadius: 235,
        alpha: 0.28,
        speed: 2.25,
        tone: 'indigo',
      });
    }
    if (ripplesRef.current.length > 12) {
      ripplesRef.current.splice(0, ripplesRef.current.length - 12);
    }
  };

  useEffect(() => {
    const stage = stageRef.current;
    const canvas = canvasRef.current;
    if (!stage || !canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let width = stage.clientWidth;
    let height = stage.clientHeight;
    let dpr = Math.min(window.devicePixelRatio || 1, 2);
    let isDark = document.documentElement.classList.contains('dark');

    const prefersReducedMotion = window.matchMedia(
      '(prefers-reduced-motion: reduce)'
    ).matches;

    const motes: Mote[] = Array.from({ length: 28 }, (_, idx) => {
      const layer = idx % 3;
      const baseRadiusX = 76 + layer * 56 + ((idx * 17) % 44);
      const baseRadiusY = 54 + layer * 38 + ((idx * 13) % 30);
      const angle = (idx / 28) * Math.PI * 2 + (idx % 5) * 0.23;
      const dir = idx % 2 === 0 ? 1 : -1;
      return {
        angle,
        baseRadiusX,
        baseRadiusY,
        speed: dir * (0.0022 + (idx % 4) * 0.0007),
        size: 1.5 + (idx % 3) * 0.65,
        kind: idx % 7 === 0 ? 'plus' : idx % 3 === 0 ? 'diamond' : 'dot',
        tone: idx % 3 === 0 ? 'indigo' : idx % 2 === 0 ? 'sky' : 'slate',
        x: width / 2 + Math.cos(angle) * baseRadiusX,
        y: height * 0.38 + Math.sin(angle) * baseRadiusY,
        vx: 0,
        vy: 0,
        alpha: 0.35 + (idx % 4) * 0.12,
      };
    });

    const syncSize = () => {
      if (!stage || !canvas) return;
      width = stage.clientWidth;
      height = stage.clientHeight;
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.max(1, Math.floor(width * dpr));
      canvas.height = Math.max(1, Math.floor(height * dpr));
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    };

    syncSize();

    const resizeObserver = new ResizeObserver(() => {
      syncSize();
      if (prefersReducedMotion) {
        drawFrame(0, true);
      }
    });
    resizeObserver.observe(stage);

    const themeObserver = new MutationObserver(() => {
      isDark = document.documentElement.classList.contains('dark');
      if (prefersReducedMotion) {
        drawFrame(0, true);
      }
    });
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class'],
    });

    const getBeaconAnchor = () => {
      const beacon = beaconRef.current;
      if (beacon && stage) {
        const sRect = stage.getBoundingClientRect();
        const bRect = beacon.getBoundingClientRect();
        if (sRect.width > 0 && bRect.width > 0) {
          return {
            x: bRect.left - sRect.left + bRect.width / 2,
            y: bRect.top - sRect.top + bRect.height / 2,
          };
        }
      }
      return { x: width / 2, y: height * 0.34 };
    };

    const getToneColor = (tone: Mote['tone'], alpha: number) => {
      if (tone === 'sky') {
        return isDark
          ? `rgba(56, 189, 248, ${alpha})`
          : `rgba(2, 132, 199, ${alpha})`;
      }
      if (tone === 'indigo') {
        return isDark
          ? `rgba(129, 140, 248, ${alpha})`
          : `rgba(99, 102, 241, ${alpha})`;
      }
      return isDark
        ? `rgba(148, 163, 184, ${alpha * 0.72})`
        : `rgba(100, 116, 139, ${alpha * 0.65})`;
    };

    let lastHeartbeat = 0;
    let rafId = 0;

    function drawFrame(now: number, staticOnly = false) {
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);

      const anchor = getBeaconAnchor();
      const draftActive = hasDraftRef.current;
      const radiusScale = draftActive ? 0.84 : 1;
      const speedScale = draftActive ? 1.55 : 1;
      const ptr = pointerRef.current;

      // Ambient breathing listening rings from the beacon center
      if (!staticOnly) {
        const ringCycle = draftActive ? 2000 : 2900;
        if (now - lastHeartbeat > ringCycle) {
          lastHeartbeat = now;
        }
        const phase = ((now % ringCycle) / ringCycle);
        const phase2 = (((now + ringCycle * 0.5) % ringCycle) / ringCycle);

        for (const p of [phase, phase2]) {
          const r = 28 + p * (draftActive ? 92 : 118);
          const ringAlpha = (1 - p) * (1 - p) * (isDark ? 0.2 : 0.16);
          ctx.beginPath();
          ctx.arc(anchor.x, anchor.y, r, 0, Math.PI * 2);
          ctx.strokeStyle = isDark
            ? `rgba(56, 189, 248, ${ringAlpha})`
            : `rgba(14, 165, 233, ${ringAlpha})`;
          ctx.lineWidth = 1.25;
          ctx.stroke();
        }
      }

      // Draw & advance interactive click ripples
      const activeRipples = ripplesRef.current;
      for (let i = activeRipples.length - 1; i >= 0; i--) {
        const rip = activeRipples[i];
        rip.radius += rip.speed;
        rip.alpha *= 0.962;

        ctx.beginPath();
        ctx.arc(rip.x, rip.y, rip.radius, 0, Math.PI * 2);
        ctx.strokeStyle =
          rip.tone === 'indigo'
            ? isDark
              ? `rgba(129, 140, 248, ${rip.alpha})`
              : `rgba(99, 102, 241, ${rip.alpha})`
            : isDark
            ? `rgba(56, 189, 248, ${rip.alpha})`
            : `rgba(14, 165, 233, ${rip.alpha})`;
        ctx.lineWidth = 1.5;
        ctx.stroke();

        if (rip.radius >= rip.maxRadius || rip.alpha <= 0.015) {
          activeRipples.splice(i, 1);
        }
      }

      // Update & render orbital motes + cursor magnetic tethers
      const nearPointer: Mote[] = [];
      for (const m of motes) {
        if (!staticOnly) {
          m.angle += m.speed * speedScale;
        }
        const targetX = anchor.x + Math.cos(m.angle) * m.baseRadiusX * radiusScale;
        const targetY = anchor.y + Math.sin(m.angle) * m.baseRadiusY * radiusScale;

        if (!staticOnly) {
          // Gentle spring back toward natural orbit
          m.vx += (targetX - m.x) * 0.036;
          m.vy += (targetY - m.y) * 0.036;

          // Subtle magnetic curiosity orbit around active pointer
          if (ptr.active) {
            const dx = ptr.x - m.x;
            const dy = ptr.y - m.y;
            const dist = Math.hypot(dx, dy);
            if (dist > 1 && dist < 165) {
              const influence = (1 - dist / 165) * 0.45;
              const desiredOrbit = 42;
              const radialForce = dist > desiredOrbit ? 1 : -0.85;
              m.vx += (dx / dist) * influence * radialForce;
              m.vy += (dy / dist) * influence * radialForce;
              // Soft tangential swirl around cursor
              m.vx += (-dy / dist) * influence * 0.42;
              m.vy += (dx / dist) * influence * 0.42;
              nearPointer.push(m);
            }
          }

          // Scatter gently from expanding click ripples
          for (const rip of activeRipples) {
            const dx = m.x - rip.x;
            const dy = m.y - rip.y;
            const dist = Math.hypot(dx, dy);
            if (dist > 1 && Math.abs(dist - rip.radius) < 22) {
              const push = rip.alpha * 1.9;
              m.vx += (dx / dist) * push;
              m.vy += (dy / dist) * push;
            }
          }

          m.vx *= 0.86;
          m.vy *= 0.86;
          m.x += m.vx;
          m.y += m.vy;
        } else {
          m.x = targetX;
          m.y = targetY;
        }
      }

      // Draw gossamer constellation tethers near cursor
      if (ptr.active && nearPointer.length > 0 && !staticOnly) {
        ctx.lineWidth = 1;
        for (let i = 0; i < nearPointer.length; i++) {
          const a = nearPointer[i];
          const distToCursor = Math.hypot(ptr.x - a.x, ptr.y - a.y);
          const cursorAlpha = Math.max(0, (1 - distToCursor / 165) * (isDark ? 0.24 : 0.18));
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(ptr.x, ptr.y);
          ctx.strokeStyle = isDark
            ? `rgba(56, 189, 248, ${cursorAlpha})`
            : `rgba(2, 132, 199, ${cursorAlpha})`;
          ctx.stroke();

          for (let j = i + 1; j < nearPointer.length; j++) {
            const b = nearPointer[j];
            const pairDist = Math.hypot(a.x - b.x, a.y - b.y);
            if (pairDist < 84) {
              const pairAlpha = (1 - pairDist / 84) * (isDark ? 0.18 : 0.14);
              ctx.beginPath();
              ctx.moveTo(a.x, a.y);
              ctx.lineTo(b.x, b.y);
              ctx.strokeStyle = isDark
                ? `rgba(129, 140, 248, ${pairAlpha})`
                : `rgba(99, 102, 241, ${pairAlpha})`;
              ctx.stroke();
            }
          }
        }
      }

      // Render each study mote
      for (const m of motes) {
        const isHighlighted =
          ptr.active && Math.hypot(ptr.x - m.x, ptr.y - m.y) < 120;
        const alpha = Math.min(0.88, m.alpha + (isHighlighted ? 0.28 : 0) + (draftActive ? 0.12 : 0));
        const fill = getToneColor(m.tone, alpha);

        ctx.save();
        ctx.translate(m.x, m.y);
        if (m.kind === 'dot') {
          ctx.beginPath();
          ctx.arc(0, 0, m.size, 0, Math.PI * 2);
          ctx.fillStyle = fill;
          ctx.fill();
        } else if (m.kind === 'diamond') {
          ctx.rotate(Math.PI / 4 + m.angle * 0.5);
          const s = m.size * 1.45;
          ctx.fillStyle = fill;
          ctx.fillRect(-s / 2, -s / 2, s, s);
        } else {
          const arm = m.size * 1.4;
          ctx.strokeStyle = fill;
          ctx.lineWidth = 1.25;
          ctx.beginPath();
          ctx.moveTo(-arm, 0);
          ctx.lineTo(arm, 0);
          ctx.moveTo(0, -arm);
          ctx.lineTo(0, arm);
          ctx.stroke();
        }
        ctx.restore();
      }
    }

    if (prefersReducedMotion) {
      drawFrame(0, true);
    } else {
      const tick = (now: number) => {
        drawFrame(now, false);
        rafId = window.requestAnimationFrame(tick);
      };
      rafId = window.requestAnimationFrame(tick);
    }

    return () => {
      if (rafId) window.cancelAnimationFrame(rafId);
      resizeObserver.disconnect();
      themeObserver.disconnect();
    };
  }, []);

  const handleStagePointerMove = (e: PointerEvent<HTMLDivElement>) => {
    const stage = stageRef.current;
    if (!stage) return;
    const rect = stage.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    pointerRef.current = { x, y, active: true };

    const beacon = beaconRef.current;
    if (
      beacon &&
      !window.matchMedia('(prefers-reduced-motion: reduce)').matches
    ) {
      const bRect = beacon.getBoundingClientRect();
      const bx = bRect.left + bRect.width / 2;
      const by = bRect.top + bRect.height / 2;
      const dx = Math.max(-1, Math.min(1, (e.clientX - bx) / 220));
      const dy = Math.max(-1, Math.min(1, (e.clientY - by) / 220));
      beacon.style.setProperty('--rx', `${(-dy * 9).toFixed(2)}deg`);
      beacon.style.setProperty('--ry', `${(dx * 9).toFixed(2)}deg`);
    }
  };

  const handleStagePointerLeave = () => {
    pointerRef.current.active = false;
    const beacon = beaconRef.current;
    if (beacon) {
      beacon.style.setProperty('--rx', '0deg');
      beacon.style.setProperty('--ry', '0deg');
      beacon.style.setProperty('--sc', '1');
    }
  };

  const handleStageClick = (e: MouseEvent<HTMLDivElement>) => {
    // Allow users to highlight/select welcome text without spawning accidental ripples
    const selection = window.getSelection();
    if (selection && selection.toString().trim().length > 0) return;

    const stage = stageRef.current;
    if (!stage) return;
    const rect = stage.getBoundingClientRect();
    spawnRipple(e.clientX - rect.left, e.clientY - rect.top, {
      tone: hasDraft ? 'indigo' : 'sky',
    });
  };

  const handleBeaconClick = (e: MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    const stage = stageRef.current;
    const beacon = beaconRef.current;
    if (stage && beacon) {
      const sRect = stage.getBoundingClientRect();
      const bRect = beacon.getBoundingClientRect();
      spawnRipple(
        bRect.left - sRect.left + bRect.width / 2,
        bRect.top - sRect.top + bRect.height / 2,
        { double: true, tone: 'sky' }
      );
      beacon.style.setProperty('--sc', '0.92');
      window.setTimeout(() => {
        beacon.style.setProperty('--sc', '1');
      }, 130);
    }
    onFocusComposer();
  };

  return (
    <div
      ref={stageRef}
      onPointerMove={handleStagePointerMove}
      onPointerLeave={handleStagePointerLeave}
      onClick={handleStageClick}
      className="relative flex flex-col items-center justify-center min-h-[380px] sm:min-h-[430px] px-4 py-14 text-center overflow-hidden"
    >
      {/* Interactive constellation & pulse-ripple canvas */}
      <canvas
        ref={canvasRef}
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 w-full h-full"
      />

      <div className="relative z-10 max-w-lg mx-auto flex flex-col items-center space-y-3.5">
        <button
          ref={beaconRef}
          type="button"
          onClick={handleBeaconClick}
          aria-label="Focus message box"
          title="Click to start typing"
          className="waiting-beacon-frame w-13 h-13 inline-flex items-center justify-center text-sky-600 dark:text-sky-400 cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-sky-600"
        >
          <BookOpen className="w-5 h-5 relative z-10" />
        </button>

        <h1
          dir="auto"
          className="font-display text-xl sm:text-[22px] font-semibold tracking-tight text-slate-900 dark:text-slate-100 text-balance"
        >
          {chatTitle}
        </h1>

        <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed max-w-md mx-auto">
          Ask a question, paste lecture screenshots, or attach study notes. Your conversation memory, notes, and textbook stay synchronized automatically for this session.
        </p>

        <div className="pt-1">
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onFocusComposer();
            }}
            className="group inline-flex items-center gap-2 px-2 py-1.5 rounded-lg text-xs font-medium text-slate-500 dark:text-slate-400 hover:text-sky-600 dark:hover:text-sky-400 transition-colors cursor-pointer focus-visible:outline-2 focus-visible:outline-sky-600"
          >
            {/* Three-bar harmonic listening cadence (not a static pulsing dot) */}
            <span
              aria-hidden="true"
              className="inline-flex items-end gap-[2.5px] h-3.5"
            >
              <span
                className="waiting-wave-bar w-[2px] h-2.5 rounded-full bg-sky-500/80 dark:bg-sky-400/80"
                style={{ animationDelay: '0ms' }}
              />
              <span
                className="waiting-wave-bar w-[2px] h-3.5 rounded-full bg-indigo-500/80 dark:bg-indigo-400/80"
                style={{ animationDelay: '180ms' }}
              />
              <span
                className="waiting-wave-bar w-[2px] h-2 rounded-full bg-sky-500/80 dark:bg-sky-400/80"
                style={{ animationDelay: '360ms' }}
              />
            </span>
            <span>
              {hasDraft
                ? 'Ready to send your message'
                : 'Waiting for your message'}
            </span>
            {!hasDraft ? (
              <span
                aria-hidden="true"
                className="waiting-caret inline-block w-[1.5px] h-3.5 rounded-full bg-sky-600 dark:bg-sky-400 -ml-0.5"
              />
            ) : null}
          </button>
        </div>
      </div>
    </div>
  );
}
