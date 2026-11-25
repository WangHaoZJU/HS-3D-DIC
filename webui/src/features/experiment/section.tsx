import { useParams } from '@tanstack/react-router'
import { SECTIONS } from '@/lib/stages'
import { Planned } from '@/components/planned'
import { CalibrationPage } from './calibration'
import { ConfigEditor } from './config'
import { ImportPage } from './import'
import { ReconstructPage } from './reconstruct'
import { ReferencePage } from './reference'
import { TemporalPage } from './temporal'

const PLANNED: Record<string, { items: string[]; note?: string }> = {
  import: {
    items: [
      '服务器文件浏览器中选择左右相机目录，探测帧数、分辨率、dtype 与有效位深（例如 16 位容器中的 12 位数据）',
      '左右配对、编号缺口与丢帧检查；触发帧对齐（两台相机预触发长度可能不同）',
      '左右联动帧查看器：缩放平移、逐帧播放、饱和区叠加',
      '灰度直方图、饱和比例、散斑尺寸、逐帧清晰度曲线（用于挑选参考帧）',
    ],
    note: '目前可以在“配置”页直接填写相机目录、帧率和时间戳文件。',
  },
  calibration: {
    items: [
      '标定板参数与示意图，先“仅检测”列出每个视图的角点数',
      '立体 RMS、基线、光轴夹角、内参及标准差',
      '逐视图 RMS 柱状图（可点选剔除后重标）、残差散点、相机与标定板位姿三维视图',
    ],
    note: '现在已经可以从右上角“运行 → 标定”提交标定任务，结果写入 calibration.h5 / .json / .png。',
  },
  reference: {
    items: [
      '在左参考图上绘制 ROI、POI 掩膜（多边形加减）与刚体参考区掩膜',
      '子区与步长变化时实时预览 POI 网格',
      '左右图 POI 叠加（按状态码或 ZNSSD 着色）、三维形貌网格与局部坐标系',
    ],
  },
  temporal: {
    items: [
      '运行中实时显示逐帧有效点比例与 ZNSSD 中位数',
      '状态码时空热力图',
      '点选 POI 查看图像轨迹与 u/v 时间曲线',
    ],
    note: '时序任务已支持进度推送、协作取消和断点续算。',
  },
  reconstruct: {
    items: [
      '刚体运动 R、t 时间曲线',
      '三维网格上的全场位移浏览与播放',
      '点选 POI 对比原始与补偿后的三分量时间序列',
    ],
  },
  vibration: {
    items: [
      '全场平均 Welch 谱与峰值表',
      '谐波拟合振幅/相位云图与 ODS 三维动画',
      'FDD 奇异值谱与振型、FRF 与相干函数、MAC 矩阵',
      '欠采样混叠提示与显著性门限',
    ],
  },
  quality: {
    items: [
      '静态噪声底（E1）云图与噪声谱',
      '不确定度传播与蒙特卡洛',
      '有效点比例、缺测比例、状态码分布汇总',
    ],
  },
  report: {
    items: [
      '一键生成 HTML 报告（可打印为 PDF）',
      '导出频率表 CSV、时间序列 CSV、振幅相位 NPZ、三维形貌 PLY',
    ],
  },
}

export function ExperimentSection() {
  const { name, section } = useParams({
    from: '/_app/experiments/$name/$section',
  })
  if (section === 'config') return <ConfigEditor name={name} />
  if (section === 'import') return <ImportPage name={name} />
  if (section === 'calibration') return <CalibrationPage name={name} />
  if (section === 'reference') return <ReferencePage name={name} />
  if (section === 'temporal') return <TemporalPage name={name} />
  if (section === 'reconstruct') return <ReconstructPage name={name} />
  const meta = SECTIONS.find((s) => s.key === section)
  const plan = PLANNED[section]
  if (!meta || !plan)
    return <p className='text-muted-foreground'>未知页面：{section}</p>
  return (
    <Planned
      icon={meta.icon}
      milestone={meta.milestone ?? ''}
      items={plan.items}
      note={plan.note}
    />
  )
}
