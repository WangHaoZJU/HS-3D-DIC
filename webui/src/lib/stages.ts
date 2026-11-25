import {
  Activity,
  Camera,
  ClipboardCheck,
  Crosshair,
  FileText,
  FolderInput,
  Grid3x3,
  Move3d,
  Settings2,
  Waves,
  type LucideIcon,
} from 'lucide-react'
import type { JobState, StageKey, StageState } from './api'

/** Pipeline steps shown in the stepper (docs/UI_Design.md §4.2). */
export const PIPELINE: { key: StageKey; label: string; section: string }[] = [
  { key: 'calibrate', label: '标定', section: 'calibration' },
  { key: 'stereo', label: '参考帧/立体', section: 'reference' },
  { key: 'temporal', label: '时序 DIC', section: 'temporal' },
  { key: 'reconstruct', label: '重建', section: 'reconstruct' },
  { key: 'rbm', label: '刚体补偿', section: 'reconstruct' },
  { key: 'analyze', label: '分析', section: 'vibration' },
  { key: 'uncertainty', label: '不确定度', section: 'quality' },
]

/** Experiment pages (sidebar order = processing order). */
export const SECTIONS: {
  key: string
  label: string
  icon: LucideIcon
  milestone?: string
  description: string
}[] = [
  {
    key: 'config',
    label: '配置',
    icon: Settings2,
    description: '实验参数、路径与 DIC 设置',
  },
  {
    key: 'import',
    label: '数据导入',
    icon: FolderInput,
    description: '导入相机导出的序列，检查位深、饱和、散斑、同步与丢帧',
  },
  {
    key: 'calibration',
    label: '标定',
    icon: Camera,
    description: 'ChArUco 检测、立体标定、逐视图残差与剔除',
  },
  {
    key: 'reference',
    label: '参考帧',
    icon: Crosshair,
    milestone: 'U3',
    description: '选择参考帧、绘制 ROI 与掩膜、立体匹配与三维形貌',
  },
  {
    key: 'temporal',
    label: '时序 DIC',
    icon: Grid3x3,
    milestone: 'U4',
    description: '固定参考时序追踪、实时有效率与状态码时空图',
  },
  {
    key: 'reconstruct',
    label: '重建',
    icon: Move3d,
    milestone: 'U4',
    description: '三角化、刚体补偿与全场位移浏览',
  },
  {
    key: 'vibration',
    label: '振动分析',
    icon: Waves,
    milestone: 'U5',
    description: '全场平均谱、谐波拟合、ODS 动画、FDD、FRF 与 MAC',
  },
  {
    key: 'quality',
    label: '质量',
    icon: ClipboardCheck,
    milestone: 'U5',
    description: '静态噪声底、不确定度传播与质量汇总',
  },
  {
    key: 'report',
    label: '报告',
    icon: FileText,
    milestone: 'U5',
    description: '生成 HTML 报告与导出 CSV / NPZ / PLY',
  },
]

/** Stages that the worker can run today, for the run menu. */
export const RUNNABLE: { key: string; label: string; hint: string }[] = [
  {
    key: 'calibrate_detect',
    label: '检测标定板角点',
    hint: '只检测，结果缓存供标定复用',
  },
  {
    key: 'calibrate',
    label: '标定',
    hint: '用缓存的角点做立体标定（需要时先检测）',
  },
  {
    key: 'stereo',
    label: '参考帧立体匹配',
    hint: '重建整个结果文件；下游结果会被清空',
  },
  { key: 'temporal', label: '时序 DIC', hint: '帧级断点续算' },
  { key: 'reconstruct', label: '三维重建', hint: '帧级断点续算' },
  { key: 'rbm', label: '刚体补偿', hint: '' },
  {
    key: 'pipeline',
    label: '立体 → 时序 → 重建 → 补偿',
    hint: '已是最新的阶段自动跳过',
  },
]

export const STAGE_STATE: Record<
  StageState,
  { label: string; dot: string; text: string }
> = {
  missing: {
    label: '未运行',
    dot: 'bg-idle/50',
    text: 'text-muted-foreground',
  },
  done: { label: '最新', dot: 'bg-ok', text: 'text-ok' },
  stale: { label: '过期', dot: 'bg-warn', text: 'text-warn' },
  partial: { label: '中断', dot: 'bg-info', text: 'text-info' },
  unknown: { label: '未知', dot: 'bg-idle', text: 'text-muted-foreground' },
  running: {
    label: '运行中',
    dot: 'bg-brand animate-pulse',
    text: 'text-brand',
  },
}

export const JOB_STATE: Record<JobState, { label: string; className: string }> =
  {
    queued: {
      label: '排队',
      className: 'bg-idle/15 text-muted-foreground border-idle/30',
    },
    running: {
      label: '运行中',
      className: 'bg-brand/15 text-brand border-brand/30',
    },
    cancelling: {
      label: '取消中',
      className: 'bg-warn/15 text-warn border-warn/30',
    },
    succeeded: { label: '完成', className: 'bg-ok/15 text-ok border-ok/30' },
    failed: { label: '失败', className: 'bg-bad/15 text-bad border-bad/30' },
    cancelled: {
      label: '已取消',
      className: 'bg-idle/15 text-muted-foreground border-idle/30',
    },
  }

export const STAGE_LABEL: Record<string, string> = {
  calibrate: '标定',
  calibrate_detect: '角点检测',
  stereo: '立体匹配',
  temporal: '时序 DIC',
  reconstruct: '重建',
  rbm: '刚体补偿',
  analyze: '振动分析',
  uncertainty: '不确定度',
  pipeline: '流水线',
  noop: '测试任务',
}

export const ActiveIcon = Activity
