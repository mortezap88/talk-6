import { useEffect, useRef, useState } from 'react';
import {
  Brain,
  Globe,
  Menu,
  Settings,
} from 'lucide-react';
import type { GeminiModel, ThinkingLevel } from '../types/talk';
import {
  normalizeThinkingLevelForModel,
  supportsXhighThinking,
} from '../utils/formatters';

interface WorkspaceRailProps {
  showSessionButton: boolean;
  onOpenSessionDrawer: () => void;
  webSearchArmed: boolean;
  onToggleWebSearch: () => void;
  geminiModel: GeminiModel;
  thinkingLevel: ThinkingLevel;
  onSelectThinkingLevel: (level: ThinkingLevel) => void;
  onOpenSettings: () => void;
}

const THINKING_OPTIONS: Array<{ id: ThinkingLevel; label: string; description: string }> = [
  { id: 'low', label: 'Low', description: 'Fastest responses' },
  { id: 'medium', label: 'Medium', description: 'Balanced reasoning' },
  { id: 'high', label: 'High', description: 'Deep reasoning' },
  { id: 'xhigh', label: 'XHigh', description: 'Maximum reasoning depth' },
];

const THINKING_LABEL: Record<ThinkingLevel, string> = {
  low: 'Low',
  medium: 'Medium',
  high: 'High',
  xhigh: 'XHigh',
};

export function WorkspaceRail({
  showSessionButton,
  onOpenSessionDrawer,
  webSearchArmed,
  onToggleWebSearch,
  geminiModel,
  thinkingLevel,
  onSelectThinkingLevel,
  onOpenSettings,
}: WorkspaceRailProps) {
  const [isThinkingOpen, setIsThinkingOpen] = useState(false);
  const thinkingContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isThinkingOpen) return;

    const handlePointerDown = (e: PointerEvent) => {
      if (
        thinkingContainerRef.current &&
        !thinkingContainerRef.current.contains(e.target as Node)
      ) {
        setIsThinkingOpen(false);
      }
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setIsThinkingOpen(false);
      }
    };

    window.addEventListener('pointerdown', handlePointerDown);
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('pointerdown', handlePointerDown);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isThinkingOpen]);

  const supportsXhigh = supportsXhighThinking(geminiModel);
  const effectiveThinkingLevel = normalizeThinkingLevelForModel(thinkingLevel, geminiModel);
  const availableThinkingOptions = supportsXhigh
    ? THINKING_OPTIONS
    : THINKING_OPTIONS.filter((option) => option.id !== 'xhigh');
  const isElevatedThinking =
    effectiveThinkingLevel === 'high' || effectiveThinkingLevel === 'xhigh';

  return (
    <div className="absolute top-3 left-3 z-30 flex flex-col items-center gap-1.5 select-none">
      {showSessionButton ? (
        <button
          type="button"
          onClick={onOpenSessionDrawer}
          aria-label="Open Chat Manager"
          aria-expanded={false}
          title="Chat Manager"
          className="h-9 w-9 inline-flex items-center justify-center rounded-lg border border-slate-200 dark:border-slate-700 bg-white/95 dark:bg-slate-900/95 backdrop-blur-xs shadow-xs text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-white transition-colors focus-visible:outline-2 focus-visible:outline-sky-600"
        >
          <Menu className="w-4 h-4" />
        </button>
      ) : null}

      <button
        type="button"
        onClick={onOpenSettings}
        aria-label="Open Settings"
        title="Settings"
        className="h-9 w-9 inline-flex items-center justify-center rounded-lg border border-slate-200 dark:border-slate-700 bg-white/95 dark:bg-slate-900/95 backdrop-blur-xs shadow-xs text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-white transition-colors shrink-0 focus-visible:outline-2 focus-visible:outline-sky-600"
      >
        <Settings className="w-4 h-4" />
      </button>

      <div className="flex flex-col items-center gap-1">
        <button
          type="button"
          onClick={onToggleWebSearch}
          aria-label="Web search"
          aria-pressed={webSearchArmed}
          title="Toggle Tavily web search for next message"
          className={`h-9 w-9 inline-flex items-center justify-center rounded-lg border backdrop-blur-xs shadow-xs transition-colors shrink-0 focus-visible:outline-2 focus-visible:outline-sky-600 ${
            webSearchArmed
              ? 'bg-sky-50 dark:bg-sky-950/80 border-sky-300 dark:border-sky-700 text-sky-700 dark:text-sky-300'
              : 'border-slate-200 dark:border-slate-700 bg-white/95 dark:bg-slate-900/95 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-white'
          }`}
        >
          <Globe className="w-4 h-4" />
        </button>

        <div ref={thinkingContainerRef} className="relative h-9 w-9 shrink-0">
          <div
            data-testid="thinking-expander"
            data-expanded={isThinkingOpen ? 'true' : 'false'}
            className={`absolute left-0 top-0 z-40 h-9 w-max inline-flex items-center rounded-lg border backdrop-blur-xs overflow-hidden transition-all duration-200 ease-out origin-left ${
              isThinkingOpen
                ? 'border-slate-200 dark:border-slate-700 bg-white/95 dark:bg-slate-900/95 shadow-md pr-1.5'
                : isElevatedThinking
                ? 'bg-sky-50 dark:bg-sky-950/80 border-sky-300 dark:border-sky-700 shadow-xs'
                : 'border-slate-200 dark:border-slate-700 bg-white/95 dark:bg-slate-900/95 shadow-xs'
            }`}
          >
            <button
              type="button"
              onClick={() => setIsThinkingOpen((prev) => !prev)}
              aria-label={`Thinking: ${THINKING_LABEL[effectiveThinkingLevel]}`}
              aria-expanded={isThinkingOpen}
              title={`Thinking: ${THINKING_LABEL[effectiveThinkingLevel]}`}
              className={`h-full w-[34px] inline-flex items-center justify-center shrink-0 transition-colors focus-visible:outline-2 focus-visible:outline-sky-600 ${
                isElevatedThinking
                  ? 'text-sky-700 dark:text-sky-300 hover:bg-sky-100/70 dark:hover:bg-sky-900/50'
                  : effectiveThinkingLevel === 'low'
                  ? 'text-slate-400 dark:text-slate-500 hover:bg-slate-50 dark:hover:bg-slate-800 hover:text-slate-700 dark:hover:text-slate-300'
                  : 'text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              <Brain className="w-4 h-4" />
            </button>

            <div
              role="radiogroup"
              aria-label="Thinking depth"
              aria-hidden={!isThinkingOpen}
              className={`grid transition-[grid-template-columns,opacity,visibility] duration-200 ease-out ${
                isThinkingOpen
                  ? 'grid-cols-[1fr] opacity-100 visible'
                  : 'grid-cols-[0fr] opacity-0 invisible pointer-events-none'
              }`}
            >
              <div className="min-w-0 overflow-hidden flex items-center gap-0.5 whitespace-nowrap">
                <span className="h-4 w-px bg-slate-200 dark:bg-slate-700 mr-1 shrink-0" />
                {availableThinkingOptions.map((option) => {
                  const selected = effectiveThinkingLevel === option.id;
                  return (
                    <button
                      key={option.id}
                      type="button"
                      role="radio"
                      aria-checked={selected}
                      tabIndex={isThinkingOpen ? 0 : -1}
                      title={option.description}
                      onClick={() => {
                        onSelectThinkingLevel(option.id);
                        setIsThinkingOpen(false);
                      }}
                      className={`h-7 px-2 rounded-md text-xs font-medium transition-colors focus-visible:outline-2 focus-visible:outline-sky-600 ${
                        selected
                          ? 'bg-sky-600 text-white shadow-xs'
                          : 'text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-white'
                      }`}
                    >
                      {option.label}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
