import { useId } from 'react';
import { Link } from '@tanstack/react-router';
import { ChevronRightIcon, LoaderCircleIcon, type LucideIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { ComputeVendorIcon, formatComputeRuntime } from '@/components/compute-vendor-icon';
import { cn } from '@/lib/utils';
import { sidebarToolState } from './status-runtime';
import type { EngineDetailLevel } from './use-engine-detail-level';

export function compactEngineName(name: string) {
  return name.replace(/\s+\([^)]*\)\s*$/, '').trim() || name;
}

export function EngineRow({
  row,
  level,
  online,
  dotClass,
  open,
  onToggle,
}: {
  row: {
    family: string;
    Icon: LucideIcon;
    detail: string;
    title?: string | null;
    runtime?: string | null;
    problem?: string | null;
    state: string;
  };
  level: EngineDetailLevel;
  online: boolean;
  dotClass: string;
  open: boolean;
  onToggle: () => void;
}) {
  const { t } = useTranslation();
  const id = useId();
  const { family, Icon, detail, title, runtime, problem, state } = row;
  const name = t('sidebarTools.' + family);
  const status = t(sidebarToolState(state, online));
  const model = title || detail;
  const loading =
    online &&
    ['engineRuntime.loading', 'preferences.loading', 'network.switching', 'common.saving'].includes(
      state,
    );
  const modelLabel = detail.includes('/')
    ? detail.slice(detail.lastIndexOf('/') + 1)
    : compactEngineName(model);
  const stateLabel = (
    <span
      role="status"
      className="inline-flex shrink-0 items-center gap-1.5 text-[11px] font-normal text-muted-foreground"
    >
      {loading ? (
        <LoaderCircleIcon
          aria-hidden="true"
          className="size-3 animate-spin motion-reduce:animate-none"
        />
      ) : (
        <span
          aria-hidden="true"
          className={cn('size-1.5 rounded-full', online ? dotClass : 'bg-muted-foreground')}
        />
      )}
      {status}
    </span>
  );
  const content = (
    <>
      <Icon className="size-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
      <span className="min-w-0 flex-1">
        <span className="flex items-center justify-between gap-2">
          <span className="truncate text-xs font-medium text-foreground" title={name}>
            {name}
          </span>
          {level === 'simple' ? (
            stateLabel
          ) : (
            <ChevronRightIcon
              aria-hidden="true"
              className={cn(
                'size-3 shrink-0 text-muted-foreground',
                open && level === 'details' ? 'rotate-90' : 'rtl:rotate-180',
              )}
            />
          )}
        </span>
        {level !== 'simple' && (
          <span className="mt-0.5 flex items-center justify-between gap-2">
            <span
              data-slot="engine-selected-model"
              className="min-w-0 truncate text-[11px] text-muted-foreground"
              title={model}
            >
              {modelLabel}
            </span>
            {stateLabel}
          </span>
        )}
      </span>
    </>
  );
  const rowClass =
    'flex min-h-11 w-full items-center gap-2 rounded-md px-1.5 py-1 text-start outline-none hover:bg-sidebar-accent/60 focus-visible:ring-2 focus-visible:ring-ring';
  return (
    <div data-slot="engine-row" aria-busy={loading} className="min-w-0">
      {level === 'details' ? (
        <button
          type="button"
          className={rowClass}
          aria-expanded={open}
          aria-controls={id}
          aria-label={`${name}: ${status}`}
          onClick={onToggle}
        >
          {content}
        </button>
      ) : (
        <Link
          to="/settings/models/$family"
          params={{ family }}
          className={rowClass}
          aria-label={`${name}: ${status}. ${t('modelSettings.change')}`}
        >
          {content}
        </Link>
      )}
      {level === 'details' && open && (
        <div
          id={id}
          data-slot="engine-diagnostics"
          className="mb-1 ms-5 space-y-1.5 border-s border-border/60 py-2 ps-3 text-[11px] leading-relaxed text-muted-foreground [overflow-wrap:anywhere]"
        >
          <p>{model}</p>
          {title && title !== detail && <code className="block">{detail}</code>}
          {runtime && (
            <p className="flex items-center gap-1.5">
              <ComputeVendorIcon runtime={runtime} className="size-3 shrink-0" />
              {formatComputeRuntime(runtime)}
            </p>
          )}
          {problem && <p>{problem}</p>}
          <Link
            to="/settings/models/$family"
            params={{ family }}
            className="inline-flex min-h-8 items-center gap-1 rounded text-foreground underline decoration-border underline-offset-4 outline-none hover:decoration-foreground focus-visible:ring-2 focus-visible:ring-ring"
          >
            {t('modelSettings.change')}
            <ChevronRightIcon className="size-3 rtl:rotate-180" aria-hidden="true" />
          </Link>
        </div>
      )}
    </div>
  );
}
