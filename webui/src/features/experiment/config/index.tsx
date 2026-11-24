import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { yaml as yamlLang } from '@codemirror/lang-yaml'
import CodeMirror from '@uiw/react-codemirror'
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  LoaderCircle,
  RotateCcw,
  Save,
} from 'lucide-react'
import { toast } from 'sonner'
import {
  api,
  qk,
  type ExperimentConfig,
  type ExperimentDetail,
  type JsonSchema,
  type ValidationResult,
} from '@/lib/api'
import { PIPELINE } from '@/lib/stages'
import { cn } from '@/lib/utils'
import { useTheme } from '@/context/theme-provider'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Skeleton } from '@/components/ui/skeleton'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { SECTIONS } from './hints'
import { SchemaFormProvider, Section, TopField } from './schema-form'

function setIn(obj: unknown, path: string[], value: unknown): unknown {
  if (path.length === 0) return value
  const [k, ...rest] = path
  const base = (obj && typeof obj === 'object' ? obj : {}) as Record<
    string,
    unknown
  >
  return { ...base, [k]: setIn(base[k], rest, value) }
}

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value)
  useEffect(() => {
    const id = setTimeout(() => setV(value), ms)
    return () => clearTimeout(id)
  }, [value, ms])
  return v
}

export function ConfigEditor({ name }: { name: string }) {
  const detail = useQuery({
    queryKey: qk.experiment(name),
    queryFn: () => api.experiment(name),
  })
  const schema = useQuery({
    queryKey: qk.schema,
    queryFn: api.schema,
    staleTime: Infinity,
  })
  if (!detail.data || !schema.data) return <Skeleton className='h-96' />
  // remount (fresh draft) whenever the stored config changes, e.g. after a save
  return (
    <ConfigForm
      key={`${name}:${detail.dataUpdatedAt}`}
      name={name}
      detail={detail.data}
      schema={schema.data}
    />
  )
}

function ConfigForm({
  name,
  detail,
  schema,
}: {
  name: string
  detail: ExperimentDetail
  schema: JsonSchema
}) {
  const qc = useQueryClient()
  const { resolvedTheme } = useTheme()
  const brokenOnDisk =
    typeof detail.config._error === 'string'
      ? (detail.config._error as string)
      : null
  const [tab, setTab] = useState<'form' | 'yaml'>(
    brokenOnDisk ? 'yaml' : 'form'
  )
  const [draft, setDraft] = useState<ExperimentConfig>(detail.config)
  const [yamlText, setYamlText] = useState<string>(detail.yaml)
  const [advanced, setAdvanced] = useState(false)

  const dirty =
    tab === 'form'
      ? JSON.stringify(draft) !== JSON.stringify(detail.config)
      : yamlText !== detail.yaml
  const payload = useMemo(
    () => (tab === 'form' ? { config: draft } : { yaml: yamlText }),
    [tab, draft, yamlText]
  )
  const debounced = useDebounced(JSON.stringify(payload), 350)
  const check = useQuery({
    queryKey: ['validate', name, debounced],
    queryFn: () => api.validate(name, JSON.parse(debounced)),
    enabled: dirty,
    staleTime: Infinity,
    meta: { silent: true },
  })
  const validating =
    dirty && (check.isFetching || debounced !== JSON.stringify(payload))
  const validation: ValidationResult | null = dirty
    ? (check.data ?? null)
    : null

  const save = useMutation({
    mutationFn: () => api.saveConfig(name, payload),
    onSuccess: (d) => {
      qc.setQueryData(qk.experiment(name), d)
      qc.invalidateQueries({ queryKey: qk.status(name) })
      qc.invalidateQueries({ queryKey: qk.experiments })
      toast.success('配置已保存')
    },
  })

  const errors = useMemo(() => {
    const m: Record<string, string> = {}
    for (const e of validation?.errors ?? []) m[e.loc.join('.')] = e.msg
    return m
  }, [validation])

  const switchTab = (t: string) => {
    if (dirty && !confirm('切换视图会放弃当前未保存的修改，继续吗？')) return
    setDraft(detail.config)
    setYamlText(detail.yaml)
    setTab(t as 'form' | 'yaml')
  }

  const sectionErrors = (fields: string[]) =>
    Object.keys(errors).filter((k) =>
      fields.some((f) => k === f || k.startsWith(f + '.'))
    ).length

  return (
    <div className='grid gap-6 xl:grid-cols-[minmax(0,1fr)_320px]'>
      <Tabs value={tab} onValueChange={switchTab} className='min-w-0'>
        <div className='mb-3 flex flex-wrap items-center gap-3'>
          <TabsList>
            <TabsTrigger value='form'>表单</TabsTrigger>
            <TabsTrigger value='yaml'>YAML</TabsTrigger>
          </TabsList>
          {tab === 'form' && (
            <div className='ms-auto flex items-center gap-2'>
              <Switch
                id='adv'
                checked={advanced}
                onCheckedChange={setAdvanced}
              />
              <Label htmlFor='adv' className='text-sm text-muted-foreground'>
                显示高级参数
              </Label>
            </div>
          )}
        </div>
        <TabsContent value='form' className='space-y-4'>
          {brokenOnDisk && (
            <div className='rounded-xl border border-bad/40 bg-bad/10 p-4 text-sm'>
              <div className='font-medium text-bad'>
                磁盘上的 experiment.yaml 无法解析，请在 YAML 视图中修复
              </div>
              <pre className='mt-2 font-mono text-xs whitespace-pre-wrap text-muted-foreground'>
                {brokenOnDisk}
              </pre>
            </div>
          )}
          {!brokenOnDisk && (
            <SchemaFormProvider
              root={schema}
              errors={errors}
              showAdvanced={advanced}
              set={(path, value) =>
                setDraft((d) => setIn(d, path, value) as ExperimentConfig)
              }
            >
              {SECTIONS.map((s) => (
                <Section
                  key={s.key}
                  title={s.title}
                  description={s.description}
                  errorCount={sectionErrors(s.fields)}
                >
                  <div className='space-y-5'>
                    {s.fields.map((f) => (
                      <TopField
                        key={f}
                        name={f}
                        value={draft as Record<string, unknown>}
                      />
                    ))}
                  </div>
                </Section>
              ))}
            </SchemaFormProvider>
          )}
        </TabsContent>
        <TabsContent value='yaml'>
          <div className='overflow-hidden rounded-xl border'>
            <CodeMirror
              value={yamlText}
              onChange={setYamlText}
              extensions={[yamlLang()]}
              theme={resolvedTheme === 'dark' ? 'dark' : 'light'}
              minHeight='480px'
              basicSetup={{ foldGutter: true, highlightActiveLine: true }}
              className='text-[13px]'
            />
          </div>
          <p className='mt-2 text-xs text-muted-foreground'>
            保存时会补全默认值并按 Pydantic 模型规范化；注释不会保留。
          </p>
        </TabsContent>
      </Tabs>

      <aside className='space-y-4 xl:sticky xl:top-20 xl:self-start'>
        <div className='rounded-xl border bg-card p-4'>
          <div className='flex items-center gap-2 text-sm font-medium'>
            {!dirty ? (
              <>
                <CheckCircle2 className='size-4 text-ok' /> 已保存
              </>
            ) : validating || !validation ? (
              <>
                <LoaderCircle className='size-4 animate-spin text-muted-foreground' />{' '}
                校验中…
              </>
            ) : validation.valid ? (
              <>
                <CheckCircle2 className='size-4 text-ok' /> 有未保存的修改 ·
                校验通过
              </>
            ) : (
              <>
                <AlertTriangle className='size-4 text-bad' />{' '}
                {validation.errors.length} 处错误
              </>
            )}
          </div>
          {validation && !validation.valid && (
            <ul className='mt-3 max-h-56 space-y-1.5 overflow-auto text-xs'>
              {validation.errors.map((e, i) => (
                <li key={i} className='rounded-md bg-bad/10 px-2 py-1.5'>
                  <code className='font-mono text-bad'>
                    {e.loc.join('.') || '(根)'}
                  </code>
                  <div className='text-muted-foreground'>{e.msg}</div>
                </li>
              ))}
            </ul>
          )}
          {validation?.valid && validation.invalidates.length > 0 && (
            <div className='mt-3 rounded-md border border-warn/30 bg-warn/10 p-2.5 text-xs'>
              <div className='font-medium text-warn'>保存后以下阶段会过期</div>
              <div className='mt-1 flex flex-wrap gap-1'>
                {validation.invalidates.map((s) => (
                  <span
                    key={s}
                    className='rounded border border-warn/30 px-1.5 py-0.5'
                  >
                    {PIPELINE.find((p) => p.key === s)?.label ?? s}
                  </span>
                ))}
              </div>
            </div>
          )}
          <div className='mt-4 flex gap-2'>
            <Button
              className='flex-1'
              disabled={
                !dirty ||
                save.isPending ||
                (validation != null && !validation.valid) ||
                validating
              }
              onClick={() => save.mutate()}
            >
              <Save /> 保存
            </Button>
            <Button
              variant='outline'
              size='icon'
              disabled={!dirty}
              onClick={() => {
                setDraft(detail.config)
                setYamlText(detail.yaml)
              }}
              aria-label='放弃修改'
              title='放弃修改'
            >
              <RotateCcw />
            </Button>
          </div>
        </div>
        <div className='space-y-2 rounded-xl border bg-card p-4 text-xs'>
          <div className='text-sm font-medium'>文件</div>
          <PathRow label='配置' value={detail.path} />
          <PathRow label='标定' value={String(draft.calibration_file ?? '—')} />
          <PathRow label='结果' value={String(draft.output_file ?? '—')} />
          <Button asChild variant='outline' size='sm' className='mt-2 w-full'>
            <a href={api.exportUrl(name)} download>
              <Download /> 导出 YAML（可提交到仓库）
            </a>
          </Button>
        </div>
      </aside>
    </div>
  )
}

function PathRow({ label, value }: { label: string; value?: string }) {
  return (
    <div className='grid grid-cols-[2.5rem_1fr] gap-2'>
      <span className='text-muted-foreground'>{label}</span>
      <span className={cn('font-mono break-all')}>{value}</span>
    </div>
  )
}
