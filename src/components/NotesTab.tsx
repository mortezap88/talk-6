import { useEffect, useRef, useState } from 'react';
import { AlertTriangle } from 'lucide-react';
import type { NoteData } from '../types/talk';
import {
  createUndoHistory,
  isRedoShortcut,
  isUndoShortcut,
  pushUndoSnapshot,
  stepRedo,
  stepUndo,
  type UndoHistory,
} from '../utils/undoHistory';
import { RichMessage } from './RichMessage';

interface NotesTabProps {
  chatId: string;
  note: NoteData;
  onSaveNote: (chatId: string, nextNote: NoteData) => Promise<void>;
}

export function NotesTab({ chatId, note, onSaveNote }: NotesTabProps) {
  const [body, setBody] = useState(note.body);
  const [saveFailed, setSaveFailed] = useState(false);
  const [isEditing, setIsEditing] = useState(!note.body.trim());

  const containerRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingBodyRef = useRef<string | null>(null);
  const activeChatRef = useRef(chatId);
  const prevChatIdRef = useRef(chatId);
  const latestBodyRef = useRef(note.body);
  const lastSavedBodyRef = useRef(note.body);
  const onSaveNoteRef = useRef(onSaveNote);
  onSaveNoteRef.current = onSaveNote;

  const historyRef = useRef<UndoHistory<string>>(createUndoHistory(80));
  const lastTypingBurstRef = useRef<{
    kind: 'insert' | 'delete';
    at: number;
  } | null>(null);
  const nativeUndoDepthRef = useRef(0);
  const nativeRedoDepthRef = useRef(0);

  const recordUndoCheckpoint = (
    prevBody: string,
    mode: 'immediate' | 'typing',
    kind: 'insert' | 'delete' = 'insert'
  ) => {
    if (mode === 'immediate') {
      lastTypingBurstRef.current = null;
      historyRef.current = pushUndoSnapshot(historyRef.current, prevBody);
      return;
    }
    const now = Date.now();
    const prevBurst = lastTypingBurstRef.current;
    const isSameBurst =
      Boolean(prevBurst) && prevBurst!.kind === kind && now - prevBurst!.at < 700;
    if (!isSameBurst) {
      historyRef.current = pushUndoSnapshot(historyRef.current, prevBody);
    }
    lastTypingBurstRef.current = { kind, at: now };
  };

  useEffect(() => {
    activeChatRef.current = chatId;
    if (prevChatIdRef.current !== chatId) {
      prevChatIdRef.current = chatId;
      historyRef.current = createUndoHistory(80);
      lastTypingBurstRef.current = null;
      nativeUndoDepthRef.current = 0;
      nativeRedoDepthRef.current = 0;
      latestBodyRef.current = note.body;
      lastSavedBodyRef.current = note.body;
      setBody(note.body);
      setIsEditing(!note.body.trim());
      setSaveFailed(false);
      pendingBodyRef.current = null;
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
      return;
    }

    // Ignore echoes from our own autosave so typing and undo history are never disrupted
    if (note.body === lastSavedBodyRef.current || note.body === latestBodyRef.current) {
      return;
    }

    // Genuine external update
    recordUndoCheckpoint(latestBodyRef.current, 'immediate');
    nativeUndoDepthRef.current = 0;
    nativeRedoDepthRef.current = 0;
    latestBodyRef.current = note.body;
    lastSavedBodyRef.current = note.body;
    setBody(note.body);
    setIsEditing(!note.body.trim());
    setSaveFailed(false);
  }, [chatId, note.body]);

  useEffect(() => {
    if (!isEditing) {
      nativeUndoDepthRef.current = 0;
      nativeRedoDepthRef.current = 0;
      return;
    }
    textareaRef.current?.focus();
    const onGlobalPointerDown = (e: PointerEvent) => {
      const target = e.target as Node | null;
      if (!containerRef.current || !target || !containerRef.current.contains(target)) {
        if ((textareaRef.current?.value ?? '').trim()) {
          setIsEditing(false);
        }
      }
    };
    window.addEventListener('pointerdown', onGlobalPointerDown, true);
    return () => window.removeEventListener('pointerdown', onGlobalPointerDown, true);
  }, [isEditing]);

  const scheduleSave = (nextBody: string, immediate = false) => {
    setSaveFailed(false);
    latestBodyRef.current = nextBody;
    pendingBodyRef.current = nextBody;
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    const targetChatId = chatId;
    const flushSave = async () => {
      timerRef.current = null;
      const bodyToSave = pendingBodyRef.current ?? nextBody;
      pendingBodyRef.current = null;
      lastSavedBodyRef.current = bodyToSave;
      try {
        await onSaveNoteRef.current(targetChatId, {
          title: '',
          body: bodyToSave,
          tags: [],
          updatedAt: Date.now(),
        });
        if (activeChatRef.current === targetChatId) {
          setSaveFailed(false);
        }
      } catch {
        if (activeChatRef.current === targetChatId) {
          setSaveFailed(true);
        }
      }
    };

    if (immediate) {
      void flushSave();
      return;
    }

    timerRef.current = setTimeout(() => {
      void flushSave();
    }, 600);
  };

  const applyUndoRedo = (direction: 'undo' | 'redo') => {
    let res =
      direction === 'undo'
        ? stepUndo(historyRef.current, latestBodyRef.current)
        : stepRedo(historyRef.current, latestBodyRef.current);
    while (res && res.nextState === latestBodyRef.current) {
      res =
        direction === 'undo'
          ? stepUndo(res.history, latestBodyRef.current)
          : stepRedo(res.history, latestBodyRef.current);
    }
    if (!res) return;
    historyRef.current = res.history;
    lastTypingBurstRef.current = null;
    nativeUndoDepthRef.current = 0;
    nativeRedoDepthRef.current = 0;
    latestBodyRef.current = res.nextState;
    setBody(res.nextState);
    setIsEditing(true);
    scheduleSave(res.nextState, true);
  };

  const applyUndoRedoRef = useRef(applyUndoRedo);
  applyUndoRedoRef.current = applyUndoRedo;

  useEffect(() => {
    const onWindowKeyDown = (e: KeyboardEvent) => {
      const undoPressed = isUndoShortcut(e);
      const redoPressed = !undoPressed && isRedoShortcut(e);
      if (!undoPressed && !redoPressed) return;

      const active = document.activeElement as HTMLElement | null;
      const isAnyEditable = Boolean(
        active &&
          (active.tagName === 'TEXTAREA' ||
            active.isContentEditable ||
            (active.tagName === 'INPUT' && (active as HTMLInputElement).type !== 'range'))
      );

      if (active && isAnyEditable && !containerRef.current?.contains(active)) {
        return;
      }

      if (active && active === textareaRef.current) {
        const isNonLatin =
          e.key.toLowerCase() !== 'z' && e.key.toLowerCase() !== 'y';
        if (undoPressed && nativeUndoDepthRef.current > 0) {
          if (isNonLatin && typeof document.execCommand === 'function') {
            e.preventDefault();
            if (document.execCommand('undo')) return;
          } else {
            return;
          }
        } else if (redoPressed && nativeRedoDepthRef.current > 0) {
          if (isNonLatin && typeof document.execCommand === 'function') {
            e.preventDefault();
            if (document.execCommand('redo')) return;
          } else {
            return;
          }
        }
      }

      if (undoPressed && historyRef.current.past.length > 0) {
        e.preventDefault();
        applyUndoRedoRef.current('undo');
      } else if (redoPressed && historyRef.current.future.length > 0) {
        e.preventDefault();
        applyUndoRedoRef.current('redo');
      }
    };

    window.addEventListener('keydown', onWindowKeyDown);
    return () => window.removeEventListener('keydown', onWindowKeyDown);
  }, []);

  useEffect(() => {
    const mountedChatId = chatId;
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
      if (pendingBodyRef.current !== null) {
        const bodyToSave = pendingBodyRef.current;
        pendingBodyRef.current = null;
        lastSavedBodyRef.current = bodyToSave;
        void onSaveNoteRef
          .current(mountedChatId, {
            title: '',
            body: bodyToSave,
            tags: [],
            updatedAt: Date.now(),
          })
          .catch(() => {});
      }
    };
  }, [chatId]);

  const showEditor = isEditing || !body.trim();

  return (
    <div
      ref={containerRef}
      className="flex-1 min-h-0 flex flex-col bg-white dark:bg-slate-900 px-6 sm:px-8 pt-14 pb-6"
    >
      {saveFailed ? (
        <div
          role="alert"
          className="mb-3 rounded-lg border border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950/40 px-3 py-2 flex items-center gap-2 text-xs text-red-700 dark:text-red-300 shrink-0"
        >
          <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
          <span>Save failed</span>
        </div>
      ) : null}

      {showEditor ? (
        <textarea
          ref={textareaRef}
          id="note-body"
          aria-label="Conversation notes"
          dir="auto"
          value={body}
          onChange={(e) => {
            const v = e.target.value;
            const prev = latestBodyRef.current;
            const inputType = (e.nativeEvent as InputEvent | undefined)?.inputType;

            if (inputType === 'historyUndo') {
              nativeUndoDepthRef.current = Math.max(0, nativeUndoDepthRef.current - 1);
              nativeRedoDepthRef.current += 1;
            } else if (inputType === 'historyRedo') {
              nativeRedoDepthRef.current = Math.max(0, nativeRedoDepthRef.current - 1);
              nativeUndoDepthRef.current += 1;
            } else {
              nativeUndoDepthRef.current += 1;
              nativeRedoDepthRef.current = 0;
              const delta = v.length - prev.length;
              const isBulk = Math.abs(delta) > 1 || (prev.length > 0 && v.length === 0);
              recordUndoCheckpoint(
                prev,
                isBulk ? 'immediate' : 'typing',
                delta < 0 ? 'delete' : 'insert'
              );
            }

            setBody(v);
            scheduleSave(v);
          }}
          onBlur={() => {
            if (body.trim()) {
              setIsEditing(false);
            }
          }}
          placeholder="Write key formulas, highlights (==term==, @@concept@@, ^^A -> B^^, %%recall%%), or notes for this chat…"
          className="doc-scroll bidi-plaintext flex-1 min-h-0 w-full bg-transparent text-[15px] text-slate-900 dark:text-slate-100 placeholder:text-slate-400 dark:placeholder:text-slate-500 leading-relaxed focus:outline-none resize-none"
        />
      ) : (
        <div
          onClick={() => {
            const selText = window.getSelection()?.toString() || '';
            if (selText.length > 0) return;
            setIsEditing(true);
          }}
          className="doc-scroll flex-1 min-h-0 overflow-y-auto rounded-lg px-2 -mx-2 py-1 hover:bg-slate-50/70 dark:hover:bg-slate-800/30 transition-colors cursor-text"
        >
          <RichMessage content={body} variant="assistant" />
        </div>
      )}
    </div>
  );
}

