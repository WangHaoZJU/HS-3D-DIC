/**
 * Form generated from the ExperimentConfig JSON Schema (Pydantic) + presentation hints.
 * Values are plain JSON; validation is done by the server (POST /validate).
 */
import { createContext, useContext, useState } from 'react'
import { ChevronDown, FolderSearch, X } from 'lucide-react'
import type { JsonSchema } from '@/lib/api'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import { FileBrowserDialog } from '@/components/file-browser'
import { hintFor, labelFor } from './hints'

type Json = unknown
type Ctx = {
  root: JsonSchema
  errors: Record<string, string>
  showAdvanced: boolean
  set: (path: string[], value: Json) => void
}
const FormCtx = createContext<Ctx | null>(null)
const useForm = () => useContext(FormCtx)!

// ------------------------------------------------------------------ schema helpers
export function resolve(
  root: JsonSchema,
  s: JsonSchema | undefined
): JsonSchema {
  let cur = s ?? {}
  for (let i = 0; i < 10 && cur.$ref; i++) {
    const name = cur.$ref.split('/').pop()!
    cur = {
      ...root.$defs?.[name],
      ...Object.fromEntries(Object.entries(cur).filter(([k]) => k !== '$ref')),
    }
  }
  return cur
}

function unwrap(
  root: JsonSchema,
  s: JsonSchema
): { inner: JsonSchema; nullable: boolean } {
  const r = resolve(root, s)
  if (r.anyOf) {
    const nonNull = r.anyOf.filter((x) => x.type !== 'null')
    const nullable = nonNull.length < r.anyOf.length
    if (nonNull.length === 1) {
      const inner = resolve(root, nonNull[0])
      return {
        inner: {
          ...inner,
          title: r.title ?? inner.title,
          default: r.default ?? inner.default,
        },
        nullable,
      }
    }
  }
  return { inner: r, nullable: false }
}

export function defaultFor(root: JsonSchema, s: JsonSchema): Json {
  const r = resolve(root, s)
  if (r.default !== undefined) return structuredClone(r.default)
  if (r.properties) {
    const out: Record<string, Json> = {}
    for (const [k, v] of Object.entries(r.properties)) {
      const d = defaultFor(root, v)
      if (d !== undefined) out[k] = d
    }
    return out
  }
  if (r.anyOf) return null
  return undefined
}

const kind = (s: JsonSchema) => (Array.isArray(s.type) ? s.type[0] : s.type)

function enumOptions(s: JsonSchema): unknown[] | null {
  if (s.enum) return s.enum
  if (s.const !== undefined) return [s.const]
  return null
}

// ------------------------------------------------------------------ public components
export function SchemaFormProvider({
  children,
  ...ctx
}: Ctx & { children: React.ReactNode }) {
  return <FormCtx value={ctx}>{children}</FormCtx>
}

/** Renders the top-level property `name` (object → group, scalar → field). */
export function TopField({
  name,
  value,
}: {
  name: string
  value: Record<string, Json>
}) {
  const { root } = useForm()
  const s = root.properties?.[name]
  if (!s) return null
  return <Node path={[name]} schema={s} value={value?.[name]} depth={0} />
}

function Node({
  path,
  schema,
  value,
  depth,
}: {
  path: string[]
  schema: JsonSchema
  value: Json
  depth: number
}) {
  const { root, showAdvanced, set } = useForm()
  const hint = hintFor(path)
  if (hint.advanced && !showAdvanced) return null
  const { inner, nullable } = unwrap(root, schema)
  if (inner.properties) {
    if (nullable && value == null) {
      return (
        <OptionalGroup
          label={labelFor(path)}
          help={hint.help}
          onEnable={() => set(path, defaultFor(root, inner) ?? {})}
        />
      )
    }
    return (
      <ObjectGroup
        path={path}
        schema={inner}
        value={(value ?? {}) as Record<string, Json>}
        depth={depth}
        onDisable={nullable ? () => set(path, null) : undefined}
      />
    )
  }
  return <Field path={path} schema={inner} nullable={nullable} value={value} />
}

function ObjectGroup({
  path,
  schema,
  value,
  depth,
  onDisable,
}: {
  path: string[]
  schema: JsonSchema
  value: Record<string, Json>
  depth: number
  onDisable?: () => void
}) {
  const { root, showAdvanced } = useForm()
  const entries = Object.entries(schema.properties ?? {}).filter(
    ([k]) => showAdvanced || !hintFor([...path, k]).advanced
  )
  if (depth > 0 && entries.length === 0) return null
  const scalars = entries.filter(([, s]) => !unwrap(root, s).inner.properties)
  const groups = entries.filter(([, s]) => unwrap(root, s).inner.properties)
  const body = (
    <div className='space-y-5'>
      {scalars.length > 0 && (
        <div className='grid gap-x-5 gap-y-4 sm:grid-cols-2'>
          {scalars.map(([k, s]) => (
            <Node
              key={k}
              path={[...path, k]}
              schema={s}
              value={value[k]}
              depth={depth + 1}
            />
          ))}
        </div>
      )}
      {groups.map(([k, s]) => (
        <Node
          key={k}
          path={[...path, k]}
          schema={s}
          value={value[k]}
          depth={depth + 1}
        />
      ))}
    </div>
  )
  if (depth === 0) {
    return (
      <div className='space-y-3'>
        {onDisable && (
          <div className='flex items-center justify-end'>
            <Button
              type='button'
              variant='ghost'
              size='sm'
              onClick={onDisable}
              className='text-muted-foreground'
            >
              <X /> 不使用此部分
            </Button>
          </div>
        )}
        {body}
      </div>
    )
  }
  return (
    <fieldset className='rounded-lg border bg-muted/20 p-4'>
      <legend className='flex items-center gap-2 px-1 text-sm font-medium'>
        {labelFor(path)}
        {onDisable && (
          <button
            type='button'
            className='text-xs font-normal text-muted-foreground hover:text-foreground'
            onClick={onDisable}
          >
            （清除）
          </button>
        )}
      </legend>
      {body}
    </fieldset>
  )
}

function OptionalGroup({
  label,
  help,
  onEnable,
}: {
  label: string
  help?: string
  onEnable: () => void
}) {
  return (
    <div className='flex items-center justify-between gap-3 rounded-lg border border-dashed p-4'>
      <div>
        <div className='text-sm font-medium'>{label}：未设置</div>
        {help && <div className='text-xs text-muted-foreground'>{help}</div>}
      </div>
      <Button type='button' variant='outline' size='sm' onClick={onEnable}>
        启用
      </Button>
    </div>
  )
}

function FieldShell({
  path,
  schema,
  children,
  wide,
}: {
  path: string[]
  schema: JsonSchema
  children: React.ReactNode
  wide?: boolean
}) {
  const { errors } = useForm()
  const hint = hintFor(path)
  const error = errors[path.join('.')]
  const id = `f-${path.join('-')}`
  const limits = [
    schema.minimum != null && `≥ ${schema.minimum}`,
    schema.exclusiveMinimum != null && `> ${schema.exclusiveMinimum}`,
    schema.maximum != null && `≤ ${schema.maximum}`,
  ].filter(Boolean)
  return (
    <div className={cn('min-w-0 space-y-1.5', wide && 'sm:col-span-2')}>
      <div className='flex items-baseline gap-2'>
        <label htmlFor={id} className='text-sm font-medium'>
          {labelFor(path)}
        </label>
        {hint.unit && (
          <span className='text-xs text-muted-foreground'>{hint.unit}</span>
        )}
        <Tooltip>
          <TooltipTrigger asChild>
            <code className='ms-auto truncate font-mono text-[10px] text-muted-foreground/60'>
              {path.join('.')}
            </code>
          </TooltipTrigger>
          <TooltipContent>
            YAML 键：{path.join('.')}
            {limits.length > 0 && <span> · {limits.join('，')}</span>}
          </TooltipContent>
        </Tooltip>
      </div>
      {children}
      {error ? (
        <p className='text-xs text-bad'>{error}</p>
      ) : (
        hint.help && (
          <p className='text-xs text-muted-foreground'>{hint.help}</p>
        )
      )}
    </div>
  )
}

function Field({
  path,
  schema,
  nullable,
  value,
}: {
  path: string[]
  schema: JsonSchema
  nullable: boolean
  value: Json
}) {
  const { set, errors } = useForm()
  const hint = hintFor(path)
  const id = `f-${path.join('-')}`
  const invalid = !!errors[path.join('.')]
  const type = kind(schema)
  const options = enumOptions(schema)
  const onChange = (v: Json) => set(path, v)

  if (type === 'boolean') {
    return (
      <FieldShell path={path} schema={schema}>
        <div className='flex h-9 items-center'>
          <Switch id={id} checked={!!value} onCheckedChange={onChange} />
        </div>
      </FieldShell>
    )
  }

  if (options) {
    return (
      <FieldShell path={path} schema={schema}>
        <Select
          value={value == null ? '' : String(value)}
          onValueChange={(v) =>
            onChange(options.find((o) => String(o) === v) ?? v)
          }
        >
          <SelectTrigger
            id={id}
            className={cn('w-full font-mono', invalid && 'border-bad')}
          >
            <SelectValue placeholder='未设置' />
          </SelectTrigger>
          <SelectContent>
            {options.map((o) => (
              <SelectItem
                key={String(o)}
                value={String(o)}
                className='font-mono'
              >
                {String(o)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </FieldShell>
    )
  }

  if (type === 'array') {
    return (
      <TupleField
        path={path}
        schema={schema}
        nullable={nullable}
        value={value as Json[] | null}
      />
    )
  }

  if (type === 'integer' || type === 'number') {
    return (
      <FieldShell path={path} schema={schema}>
        <NumberInput
          id={id}
          value={value as number | null}
          integer={type === 'integer'}
          nullable={nullable}
          invalid={invalid}
          onChange={onChange}
        />
      </FieldShell>
    )
  }

  // string
  if (schema.format === 'path' || hint.pick) {
    return (
      <FieldShell path={path} schema={schema} wide>
        <PathInput
          id={id}
          value={value as string | null}
          nullable={nullable}
          invalid={invalid}
          onChange={onChange}
          pick={hint.pick ?? 'file'}
          root={hint.root ?? 'data'}
        />
      </FieldShell>
    )
  }
  if (hint.multiline) {
    return (
      <FieldShell path={path} schema={schema} wide>
        <Textarea
          id={id}
          rows={2}
          value={(value as string) ?? ''}
          onChange={(e) => onChange(e.target.value)}
        />
      </FieldShell>
    )
  }
  return (
    <FieldShell path={path} schema={schema}>
      <Input
        id={id}
        value={(value as string) ?? ''}
        readOnly={hint.readOnly}
        placeholder={nullable ? '未设置' : undefined}
        className={cn(
          hint.readOnly && 'bg-muted/50 font-mono',
          invalid && 'border-bad'
        )}
        onChange={(e) =>
          onChange(nullable && e.target.value === '' ? null : e.target.value)
        }
      />
    </FieldShell>
  )
}

function NumberInput({
  id,
  value,
  nullable,
  invalid,
  onChange,
  placeholder,
  className,
}: {
  className?: string
  id?: string
  value: number | null
  integer?: boolean
  nullable: boolean
  invalid?: boolean
  onChange: (v: Json) => void
  placeholder?: string
}) {
  // keep the raw text while typing so "1e-" or "0." do not get rewritten
  const [text, setText] = useState<string | null>(null)
  const shown = text ?? (value == null ? '' : String(value))
  return (
    <Input
      id={id}
      inputMode='decimal'
      value={shown}
      placeholder={placeholder ?? (nullable ? '未设置' : undefined)}
      className={cn('tabular font-mono', invalid && 'border-bad', className)}
      onChange={(e) => {
        const t = e.target.value
        setText(t)
        if (t.trim() === '') return onChange(nullable ? null : '')
        const n = Number(t)
        // keep non-numeric text so the server reports a type error
        onChange(Number.isFinite(n) ? n : t)
      }}
      onBlur={() => setText(null)}
    />
  )
}

function TupleField({
  path,
  schema,
  nullable,
  value,
}: {
  path: string[]
  schema: JsonSchema
  nullable: boolean
  value: Json[] | null
}) {
  const { set, root, errors } = useForm()
  const hint = hintFor(path)
  const items = resolve(root, schema.items)
  const integer = kind(items) === 'integer'
  const invalid = !!errors[path.join('.')]
  const labels =
    hint.tuple ??
    (value ?? (schema.default as Json[]) ?? []).map((_, i) => String(i))
  if (value == null) {
    return (
      <FieldShell path={path} schema={schema}>
        <div className='flex h-9 items-center gap-2'>
          <span className='text-sm text-muted-foreground'>未设置</span>
          <Button
            type='button'
            size='sm'
            variant='outline'
            onClick={() =>
              set(
                path,
                structuredClone(
                  hint.enableDefault ?? schema.default ?? labels.map(() => 0)
                )
              )
            }
          >
            设置
          </Button>
        </div>
      </FieldShell>
    )
  }
  return (
    <FieldShell path={path} schema={schema} wide={labels.length > 2}>
      <div className='flex items-center gap-2'>
        {labels.map((lab, i) => (
          <div key={i} className='relative min-w-0 flex-1'>
            <span className='pointer-events-none absolute start-2 top-1/2 -translate-y-1/2 text-[10px] text-muted-foreground'>
              {lab}
            </span>
            <NumberInput
              value={(value[i] as number) ?? null}
              integer={integer}
              nullable={false}
              invalid={invalid}
              onChange={(v) => {
                const next = [...value]
                next[i] = v
                set(path, next)
              }}
              placeholder=''
              className='ps-7'
            />
          </div>
        ))}
        {nullable && (
          <Button
            type='button'
            size='icon'
            variant='ghost'
            className='size-8 shrink-0'
            onClick={() => set(path, null)}
            aria-label='清除'
          >
            <X className='size-4' />
          </Button>
        )}
      </div>
    </FieldShell>
  )
}

function PathInput({
  id,
  value,
  nullable,
  invalid,
  onChange,
  pick,
  root,
}: {
  id: string
  value: string | null
  nullable: boolean
  invalid: boolean
  onChange: (v: Json) => void
  pick: 'dir' | 'file'
  root: string
}) {
  const [open, setOpen] = useState(false)
  return (
    <div className='flex gap-2'>
      <Input
        id={id}
        value={value ?? ''}
        placeholder={
          nullable ? '未设置' : pick === 'dir' ? '目录路径' : '文件路径'
        }
        className={cn('font-mono text-xs', invalid && 'border-bad')}
        onChange={(e) =>
          onChange(nullable && e.target.value === '' ? null : e.target.value)
        }
      />
      <Button
        type='button'
        variant='outline'
        size='icon'
        onClick={() => setOpen(true)}
        aria-label='浏览'
      >
        <FolderSearch />
      </Button>
      <FileBrowserDialog
        open={open}
        onOpenChange={setOpen}
        mode={pick}
        initialRoot={root}
        onPick={(p) => onChange(p)}
      />
    </div>
  )
}

/** Section card with a collapsible body. */
export function Section({
  title,
  description,
  children,
  defaultOpen = true,
  errorCount = 0,
}: {
  title: string
  description?: string
  children: React.ReactNode
  defaultOpen?: boolean
  errorCount?: number
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <Collapsible
      open={open}
      onOpenChange={setOpen}
      className='rounded-xl border bg-card'
    >
      <CollapsibleTrigger className='flex w-full items-center gap-3 px-5 py-4 text-start'>
        <div className='min-w-0 flex-1'>
          <div className='flex items-center gap-2 font-medium'>
            {title}
            {errorCount > 0 && (
              <span className='rounded-full bg-bad/15 px-1.5 text-xs text-bad'>
                {errorCount}
              </span>
            )}
          </div>
          {description && (
            <div className='mt-0.5 text-xs text-muted-foreground'>
              {description}
            </div>
          )}
        </div>
        <ChevronDown
          className={cn(
            'size-4 text-muted-foreground transition-transform',
            open && 'rotate-180'
          )}
        />
      </CollapsibleTrigger>
      <CollapsibleContent className='CollapsibleContent'>
        <div className='border-t px-5 py-5'>{children}</div>
      </CollapsibleContent>
    </Collapsible>
  )
}
