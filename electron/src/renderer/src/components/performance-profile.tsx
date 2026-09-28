import { Slider } from '@base-ui/react/slider';
import { DirectionProvider } from '@base-ui/react/direction-provider';
import { Link } from '@tanstack/react-router';
import {
  CrownIcon,
  GaugeIcon,
  SparkleIcon,
  SparklesIcon,
  WandSparklesIcon,
  InfoIcon,
  ChevronRightIcon,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useId, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { toast } from 'sonner';
import { apiJson, describeError } from '@/lib/api/client';
import { useBackendStatus } from '@/hooks/use-backend-status';
import { useDictationSelection } from '@/hooks/use-dictation-selection';
import { engineFamilyState, useEngines } from '@/hooks/use-engines';
import { Button, buttonVariants } from '@/components/ui/button';
import { useAppActivities } from '@/lib/app-activity';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/popover';
import { LiveDeviceUsage } from '@/components/live-device-usage';
import { cn } from '@/lib/utils';
import {
  performanceTiers,
  usePerformanceProfile,
  type PerformanceFamily,
  type PerformanceChoice,
  type PerformanceProfileState,
} from '@/hooks/use-performance-profile';

export function PerformanceProfile({
  family = null,
  variant = 'compact',
  onApplied,
}: {
  family?: PerformanceFamily | null;
  variant?: 'compact' | 'settings';
  onApplied?: (state: PerformanceProfileState) => void;
}) {
  const { t, i18n } = useTranslation();
  const profile = usePerformanceProfile();
  const activities = useAppActivities();
  const backend = useBackendStatus();
  const engines = useEngines();
  const dictation = useDictationSelection();
  const batch = useQuery({
    queryKey: ['batch-jobs', 'active'],
    enabled: backend.stage === 'ready',
    queryFn: ({ signal }) => apiJson<unknown[]>('/batch/jobs?status=active&limit=100', { signal }),
    staleTime: 1_000,
    refetchInterval: (query) => (query.state.data?.length ? 1_000 : 15_000),
  });
  const [failed, setFailed] = useState<PerformanceChoice | null>(null);
  const groupId = useId();
  const [draft, setDraft] = useState<number | null>(null);
  const [hardwareOpen, setHardwareOpen] = useState(false);
  const busy =
    profile.isSaving ||
    backend.stage !== 'ready' ||
    batch.isPending ||
    batch.isError ||
    Boolean(batch.data?.length) ||
    Object.values(activities).some((count) => count > 0);
  const choices: readonly PerformanceChoice[] = family
    ? performanceTiers
    : [...performanceTiers, 'auto'];
  const selected = family ? profile.data?.effective[family] : profile.data?.global;
  const applicable = profile.data?.applicable_families ?? profile.data?.implemented_families ?? [];
  const supported = family === null ? true : applicable.includes(family);
  const disabled = busy || !profile.data || !supported;
  const selectedIndex = selected ? choices.indexOf(selected) : -1;
  const position = draft ?? Math.max(0, selectedIndex);
  const TierIcon = [GaugeIcon, SparkleIcon, SparklesIcon, CrownIcon, WandSparklesIcon][position];
  const accent = [
    'var(--muted-foreground)',
    'color-mix(in oklab, var(--primary) 72%, var(--foreground))',
    'var(--primary)',
    'color-mix(in oklab, var(--primary) 82%, white)',
  ][Math.min(position, 3)];
  const glow = disabled
    ? 'none'
    : [
        'inset 0 1px 0 rgb(255 255 255 / 10%)',
        'inset 0 1px 0 rgb(255 255 255 / 18%), 0 0 8px color-mix(in oklab, var(--primary) 16%, transparent)',
        'inset 0 1px 0 rgb(255 255 255 / 30%), 0 0 16px color-mix(in oklab, var(--primary) 34%, transparent)',
        'inset 0 1px 0 rgb(255 255 255 / 36%), 0 0 20px color-mix(in oklab, var(--primary) 46%, transparent)',
      ][Math.min(position, 3)];
  const thumbGlow = disabled
    ? 'none'
    : [
        '0 1px 3px rgb(0 0 0 / 20%)',
        '0 2px 6px rgb(0 0 0 / 22%), 0 0 7px color-mix(in oklab, var(--primary) 16%, transparent)',
        '0 3px 9px rgb(0 0 0 / 24%), 0 0 13px color-mix(in oklab, var(--primary) 34%, transparent)',
        '0 4px 11px rgb(0 0 0 / 26%), 0 0 17px color-mix(in oklab, var(--primary) 48%, transparent)',
      ][Math.min(position, 3)];
  const choose = async (tier: PerformanceChoice) => {
    if (disabled) {
      setDraft(null);
      return;
    }
    setDraft(choices.indexOf(tier));
    setFailed(null);
    try {
      const applied = await profile.setTier({ tier, family });
      onApplied?.(applied);
      toast.success(
        t('performanceProfile.applied', {
          tier: t('performanceProfile.' + tier),
        }),
      );
    } catch (error) {
      setFailed(tier);
      toast.error(describeError(error));
    } finally {
      setDraft(null);
    }
  };
  const plan = profile.data?.plan;
  const hardwareSummary = profile.isSaving
    ? t('common.saving')
    : plan
      ? (selected === 'auto' ? t('performanceProfile.' + plan.resolved) + ' · ' : '') +
        t('performanceHardware.' + (plan.status === 'limited' ? 'reason.memory' : plan.status))
      : '';
  const selection = family ? profile.data?.selections?.[family] : null;
  const target = family ? profile.data?.targets?.[family] : null;
  const legacyFamily =
    family === 'tts' || family === 'asr' || family === 'llm'
      ? engineFamilyState(engines.data, family)
      : null;
  const legacyModel =
    family === 'dictation'
      ? dictation.data?.model?.label || dictation.data?.model_id
      : legacyFamily?.active_model || target?.model || target?.engine || legacyFamily?.active;
  const selectedModel = selection?.label || selection?.model || selection?.engine || legacyModel;
  const selectedEngine = selection?.engine || legacyFamily?.active || target?.engine;
  const requiredEngine = target?.engine
    ? {
        'faster-whisper': 'Faster-Whisper',
        'sherpa-onnx': 'Sherpa-ONNX',
        pyannote: 'pyannote',
        'audiocpp-sortformer': 'Sortformer',
        nllb: 'NLLB-200',
      }[target.engine] || target.engine
    : null;
  const targetMetric =
    family === 'tts' && target?.steps
      ? `${target.steps} ${t('clone.steps')}`
      : family === 'asr' && target?.beam_size
        ? `×${target.beam_size}`
        : family === 'dictation' && target?.max_active_paths
          ? `×${target.max_active_paths}`
          : family === 'translation' && target?.num_beams
            ? `×${target.num_beams}`
            : null;
  if (variant === 'settings') {
    return (
      <div className="w-full min-w-0 @2xl:w-[min(100%,34rem)]">
        <div
          role="radiogroup"
          aria-labelledby={groupId}
          aria-busy={profile.isSaving}
          className={cn(
            'grid min-w-0 grid-cols-2 gap-1 rounded-xl border border-border/60 bg-muted/45 p-1 sm:grid-flow-col sm:auto-cols-fr',
            disabled && 'opacity-50',
          )}
        >
          <span id={groupId} className="sr-only">
            {t(supported ? 'performanceProfile.title' : 'modelSettings.unavailable')}
          </span>
          {choices.map((tier, index) => {
            const Icon = [GaugeIcon, SparkleIcon, SparklesIcon, CrownIcon, WandSparklesIcon][index];
            const active = supported && position === index;
            return (
              <button
                key={tier}
                type="button"
                role="radio"
                aria-checked={active}
                data-selected={active}
                disabled={disabled}
                onClick={() => void choose(tier)}
                className={cn(
                  'performance-tier-button inline-flex h-9 min-w-0 items-center justify-center gap-1.5 rounded-lg px-2 text-xs font-medium text-muted-foreground outline-none transition-[background-color,color,box-shadow] duration-150 hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50 motion-reduce:transition-none',
                  active && 'text-foreground',
                )}
              >
                <Icon
                  className="size-3.5 shrink-0"
                  style={active ? { color: accent } : undefined}
                  aria-hidden="true"
                />
                <span className="truncate">{t('performanceProfile.' + tier)}</span>
              </button>
            );
          })}
        </div>
        {family && (
          <div className="mt-2 flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1 px-1 text-xs text-muted-foreground">
            {supported && selectedModel ? (
              <>
                <span className="truncate font-medium text-foreground/80">{selectedModel}</span>
                {selectedEngine && selectedEngine !== selectedModel && (
                  <span className="truncate">{selectedEngine}</span>
                )}
                {targetMetric && <span className="shrink-0 tabular-nums">{targetMetric}</span>}
              </>
            ) : (
              <>
                <span>
                  {requiredEngine
                    ? t('performanceProfile.requiresEngine', { engine: requiredEngine })
                    : t('performanceProfile.noTarget')}
                </span>
                <Link
                  to="/settings/models/$family"
                  params={{ family }}
                  className={buttonVariants({ variant: 'ghost', size: 'xs' })}
                >
                  {t('modelSettings.models')}
                </Link>
              </>
            )}
          </div>
        )}
        {(failed || profile.isError || batch.isError) && (
          <div role="alert" className="mt-1 text-xs text-destructive">
            {t('common.error')}
            <Button
              size="xs"
              variant="ghost"
              disabled={profile.isSaving}
              onClick={() => {
                if (batch.isError) void batch.refetch();
                else if (failed && !busy) void choose(failed);
                else void profile.refetch();
              }}
            >
              {t('common.retry')}
            </Button>
          </div>
        )}
      </div>
    );
  }
  return (
    <div className="min-w-0 py-1">
      <div className="mb-1 flex items-center justify-between gap-2 px-1 text-[11px] text-muted-foreground">
        <span>{t('performanceProfile.title')}</span>
        <span className="font-medium text-foreground">
          {t('performanceProfile.' + choices[position])}
        </span>
      </div>
      <span id={groupId} className="sr-only">
        {t(supported ? 'performanceProfile.title' : 'modelSettings.unavailable')}
      </span>
      <DirectionProvider direction={i18n.dir(i18n.language)}>
        <Slider.Root
          min={0}
          max={choices.length - 1}
          step={1}
          largeStep={1}
          value={position}
          disabled={disabled}
          aria-busy={profile.isSaving}
          onValueChange={(value) => setDraft(value)}
          onValueCommitted={(value) => void choose(choices[value])}
          className={`w-full min-w-0 ${disabled ? 'opacity-50' : ''}`}
        >
          <Slider.Control className="relative flex h-9 w-full min-w-0 touch-none select-none items-center px-4">
            <span
              aria-hidden="true"
              className="pointer-events-none absolute inset-x-0 inset-y-1 rounded-full border border-border/60 transition-[background-color] duration-200 motion-reduce:transition-none"
              style={{
                backgroundColor: disabled
                  ? 'var(--muted)'
                  : `color-mix(in oklab, ${accent} ${[5, 9, 14, 20][Math.min(position, 3)]}%, var(--muted))`,
              }}
            />
            <span
              aria-hidden="true"
              className="pointer-events-none absolute inset-y-1 start-0 rounded-full transition-[background,box-shadow,opacity] duration-200 motion-reduce:transition-none"
              style={{
                width: `calc(1rem + (100% - 2rem) * ${position / (choices.length - 1)})`,
                background: `linear-gradient(110deg, color-mix(in oklab, ${accent} 65%, var(--muted)), ${accent} 75%, color-mix(in oklab, ${accent} ${position >= 2 ? 82 : 95}%, white))`,
                boxShadow: glow,
              }}
            />
            <Slider.Track className="relative h-7 w-full min-w-0">
              <div
                aria-hidden="true"
                className="pointer-events-none absolute inset-0 flex items-center justify-between"
              >
                {choices.map((tier, index) => (
                  <span
                    key={tier}
                    title={t('performanceProfile.' + tier)}
                    className={`size-1 shrink-0 rounded-full ${index <= position ? 'bg-primary-foreground/50' : 'bg-muted-foreground/60'}`}
                  />
                ))}
              </div>
              <Slider.Thumb
                aria-labelledby={groupId}
                aria-describedby={groupId + '-help'}
                getAriaValueText={(_formatted, value) => t('performanceProfile.' + choices[value])}
                className="absolute top-1/2 z-10 grid size-7 place-items-center rounded-full border border-background/40 bg-foreground text-background outline-none ring-ring/40 transition-[box-shadow,transform] hover:scale-105 focus-within:ring-4 data-dragging:scale-105 motion-reduce:transition-none"
                style={{ boxShadow: thumbGlow }}
              >
                <TierIcon className="size-3.5" aria-hidden="true" />
              </Slider.Thumb>
            </Slider.Track>
          </Slider.Control>
        </Slider.Root>
      </DirectionProvider>
      {plan && !family && (
        <Popover open={hardwareOpen} onOpenChange={setHardwareOpen}>
          <PopoverTrigger
            render={
              <button
                type="button"
                aria-label={hardwareSummary}
                className="mt-0.5 flex min-h-7 w-full items-center gap-1.5 rounded px-1 text-start text-[11px] text-muted-foreground outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
              />
            }
          >
            <span className="min-w-0 flex-1" role="status">
              {hardwareSummary}
            </span>
            <InfoIcon className="size-3 shrink-0" aria-hidden="true" />
          </PopoverTrigger>
          <PopoverContent
            side="right"
            className="max-h-[min(42rem,80dvh)] w-80 max-w-[calc(100vw-1rem)] space-y-3 overflow-y-auto p-4 text-xs"
          >
            <div>
              <p className="font-medium">
                {t('performanceProfile.max') +
                  ' · ' +
                  t(
                    'performanceHardware.' +
                      (plan.max_status === 'limited' ? 'reason.memory' : plan.max_status),
                  )}
              </p>
              <p className="mt-1 text-muted-foreground">{t('performanceHardware.voiceFirst')}</p>
            </div>
            <LiveDeviceUsage open={hardwareOpen} />
            <div className="-mx-1">
              {(['tts', 'asr', 'translation', 'dictation', 'diarisation'] as const).map((name) => {
                const entry = plan.families[name];
                return (
                  <Link
                    key={name}
                    to="/settings/models/$family"
                    params={{ family: name }}
                    className="group flex min-h-7 items-center justify-between gap-3 rounded-md px-1 py-1 outline-none hover:bg-muted/60 focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    <span>{t('sidebarTools.' + name)}</span>
                    <span className="flex items-center gap-1.5 text-end text-[11px] text-muted-foreground">
                      {t('performanceHardware.reason.' + entry.reason)}
                      <ChevronRightIcon
                        className="size-3 shrink-0 opacity-50 group-hover:opacity-100 rtl:rotate-180"
                        aria-hidden="true"
                      />
                    </span>
                  </Link>
                );
              })}
            </div>
            <details className="group/details border-t pt-1 text-muted-foreground">
              <summary className="flex min-h-8 cursor-pointer list-none items-center justify-between gap-2 rounded outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring [&::-webkit-details-marker]:hidden">
                {t('sidebarTools.details')}
                <ChevronRightIcon
                  className="size-3.5 shrink-0 group-open/details:rotate-90 rtl:rotate-180"
                  aria-hidden="true"
                />
              </summary>
              <div className="space-y-2 pt-1 leading-relaxed">
                <p className="text-foreground/80 tabular-nums">
                  {t('performanceHardware.specs', {
                    ram:
                      plan.hardware.ram_gb?.toLocaleString(i18n.language, {
                        maximumFractionDigits: 0,
                      }) ?? '—',
                    cores: plan.hardware.cpu_threads,
                  })}
                  {plan.hardware.vram_gb != null &&
                    ' · ' +
                      t('performanceHardware.vram', {
                        memory: plan.hardware.vram_gb.toLocaleString(i18n.language, {
                          maximumFractionDigits: 0,
                        }),
                      })}
                </p>
                <p>{t('performanceHardware.priority')}</p>
                <p>{t('performanceHardware.estimate')}</p>
              </div>
            </details>
          </PopoverContent>
        </Popover>
      )}
      <p id={groupId + '-help'} className="sr-only">
        {t(
          !supported
            ? 'performanceProfile.noTarget'
            : busy
              ? 'sidebarTools.wait'
              : 'sidebarTools.speedHelp',
        )}
      </p>
      {(failed || profile.isError || batch.isError) && (
        <div role="alert" className="text-xs text-destructive">
          {t('common.error')}
          <Button
            size="xs"
            variant="ghost"
            disabled={profile.isSaving}
            onClick={() => {
              if (batch.isError) void batch.refetch();
              else if (failed && !busy) void choose(failed);
              else void profile.refetch();
            }}
          >
            {t('common.retry')}
          </Button>
        </div>
      )}
    </div>
  );
}
