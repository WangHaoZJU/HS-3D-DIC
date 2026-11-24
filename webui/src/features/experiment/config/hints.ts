/**
 * UI hints for the ExperimentConfig JSON Schema: Chinese labels, units, help, grouping.
 * The schema (from the server's Pydantic model) stays the single source of truth for
 * types, limits, enums and defaults; this file only adds presentation (UI_Design §6.3).
 */

export type Hint = {
  label?: string
  unit?: string
  help?: string
  advanced?: boolean
  /** path picker: what to pick and where to start */
  pick?: 'dir' | 'file'
  root?: 'data' | 'results' | 'workspace'
  /** fixed tuple length and per-item labels */
  tuple?: string[]
  multiline?: boolean
  readOnly?: boolean
  /** default when a nullable object/array is switched on */
  enableDefault?: unknown
}

export const SECTIONS: {
  key: string
  title: string
  description: string
  fields: string[]
}[] = [
  {
    key: 'basic',
    title: '基本信息',
    description: '名称即工作区目录名，不能在这里修改',
    fields: ['name', 'description'],
  },
  {
    key: 'cameras',
    title: '相机与图像序列',
    description:
      '左右相机导出的图像目录、帧率与同步。坐标统一换算到标定使用的全幅传感器坐标。',
    fields: ['cameras'],
  },
  {
    key: 'frames',
    title: '帧范围与参考帧',
    description: '参考帧应选在运动模糊最小的时刻（例如激振前的静止帧）',
    fields: ['frames', 'reference_frame'],
  },
  {
    key: 'region',
    title: '测量区域',
    description: 'ROI 与掩膜。U3 起可以直接在参考图上绘制掩膜。',
    fields: ['roi', 'poi_mask', 'rbm_mask'],
  },
  {
    key: 'files',
    title: '标定与结果文件',
    description:
      '相对路径分别按 data_root / results_root 解析；新建实验默认放在工作区目录',
    fields: ['calibration_file', 'output_file'],
  },
  {
    key: 'calibration',
    title: 'ChArUco 标定',
    description: '标定图像与标定板参数；不需要重新标定时可以关闭',
    fields: ['calibration'],
  },
  {
    key: 'dic',
    title: 'DIC 参数',
    description: '子区、插值、求解器，以及立体与时序匹配',
    fields: ['dic'],
  },
  {
    key: 'recon',
    title: '三维重建',
    description: '叶片局部坐标系 (c, s, n)',
    fields: ['recon'],
  },
]

export const HINTS: Record<string, Hint> = {
  name: { label: '名称', readOnly: true },
  description: { label: '描述', multiline: true },

  cameras: { label: '相机' },
  'cameras.*': { label: '相机' },
  'cameras.*.path': { label: '图像目录', pick: 'dir', root: 'data' },
  'cameras.*.pattern': {
    label: '文件匹配',
    help: '例如 *.tif；按自然排序读取',
  },
  'cameras.*.sensor_origin': {
    label: '录制窗口原点',
    unit: 'px',
    tuple: ['x', 'y'],
    help: '相机为了高帧率常只录制传感器的一部分；这里填该窗口在全幅传感器上的左上角坐标',
  },
  'cameras.*.timestamp_path': {
    label: '时间戳文件',
    pick: 'file',
    root: 'data',
    help: '可选；.npy / .csv / 文本，单位 s。没有时按帧率生成',
  },
  'cameras.frame_rate_hz': { label: '帧率', unit: 'fps' },
  'cameras.max_sync_delta_us': {
    label: '同步偏差上限',
    unit: 'µs',
    help: '左右时间戳配对允许的最大差值；留空表示不检查',
    advanced: true,
  },

  frames: { label: '帧范围' },
  'frames.start': { label: '起始帧' },
  'frames.stop': { label: '结束帧（不含）', help: '留空表示到序列末尾' },
  'frames.step': { label: '步长' },
  reference_frame: { label: '参考帧' },

  roi: {
    label: 'ROI',
    unit: 'px',
    tuple: ['x0', 'y0', '宽', '高'],
    help: '左图坐标中的 POI 区域；留空表示整幅图',
    enableDefault: [0, 0, 512, 512],
  },
  poi_mask: {
    label: 'POI 掩膜',
    pick: 'file',
    root: 'workspace',
    help: '与左图同尺寸的图像，非零处布置测点',
  },
  rbm_mask: {
    label: '刚体参考区掩膜',
    pick: 'file',
    root: 'workspace',
    help: '夹具上的静止散斑块；未设置时用全部有效点估计刚体运动',
  },

  calibration_file: {
    label: '标定文件',
    pick: 'file',
    root: 'workspace',
    help: '标定阶段写入，立体阶段读取（.h5 或 .json）',
  },
  output_file: {
    label: '结果文件',
    pick: 'file',
    root: 'workspace',
    help: 'HDF5 结果（schema v1）',
  },

  calibration: { label: '标定设置' },
  'calibration.board': { label: '标定板' },
  'calibration.board.squares_x': { label: '横向方格数' },
  'calibration.board.squares_y': { label: '纵向方格数' },
  'calibration.board.square_mm': { label: '方格边长', unit: 'mm' },
  'calibration.board.marker_mm': { label: '码边长', unit: 'mm' },
  'calibration.board.dictionary': { label: 'ArUco 字典' },
  'calibration.board.legacy_pattern': {
    label: '旧版图案',
    advanced: true,
    help: 'OpenCV 4.6 之前生成的偶数行板',
  },
  'calibration.*': { label: '标定图像' },
  'calibration.*.path': { label: '图像目录', pick: 'dir', root: 'data' },
  'calibration.*.pattern': { label: '文件匹配' },
  'calibration.*.sensor_origin': {
    label: '录制窗口原点',
    unit: 'px',
    tuple: ['x', 'y'],
    advanced: true,
  },
  'calibration.*.timestamp_path': {
    label: '时间戳文件',
    pick: 'file',
    root: 'data',
    advanced: true,
  },
  'calibration.distortion': { label: '畸变模型' },
  'calibration.refine_stereo_intrinsics': { label: '双目联合精化内参' },
  'calibration.outlier_factor': {
    label: '离群视图阈值',
    unit: '× 中位 RMS',
    advanced: true,
  },
  'calibration.min_views': { label: '最少视图数', advanced: true },
  'calibration.min_corners': { label: '每视图最少角点', advanced: true },
  'calibration.roi_origin': {
    label: '标定图裁剪原点',
    unit: 'px',
    tuple: ['x', 'y'],
    advanced: true,
  },
  'calibration.sensor_size': {
    label: '全幅传感器尺寸',
    unit: 'px',
    tuple: ['宽', '高'],
    advanced: true,
    enableDefault: [1280, 1024],
  },

  dic: { label: 'DIC' },
  'dic.backend': {
    label: '计算后端',
    help: 'auto：有 CUDA 用 cuda，否则 numba',
  },
  'dic.poi': { label: '测点网格' },
  'dic.poi.subset_size': {
    label: '子区边长',
    unit: 'px',
    help: '奇数；B03：31 px 时噪声与空间分辨率较平衡',
  },
  'dic.poi.step': { label: '网格步长', unit: 'px' },
  'dic.interpolation': { label: '插值' },
  'dic.interpolation.order': {
    label: 'B 样条阶数',
    help: '五次偏差约为三次的 1/4，吞吐约 0.7 倍',
  },
  'dic.solver': { label: 'IC-GN 求解器' },
  'dic.solver.max_iterations': { label: '最大迭代次数', advanced: true },
  'dic.solver.convergence_threshold_px': {
    label: '收敛阈值',
    unit: 'px',
    advanced: true,
  },
  'dic.solver.min_gray_std': {
    label: '最小灰度标准差',
    unit: 'GL',
    help: '低于此值记为 LOW_TEXTURE',
    advanced: true,
  },
  'dic.solver.max_condition': { label: 'Hessian 条件数上限', advanced: true },
  'dic.stereo': { label: '立体匹配 L0 ↔ R0' },
  'dic.stereo.shape_order': {
    label: '形函数阶数',
    help: '二阶能吸收两相机间的透视差，默认',
  },
  'dic.stereo.initialization': { label: '初值方法' },
  'dic.stereo.reliability_guided': { label: '可靠性引导传播 (RG-DIC)' },
  'dic.stereo.max_znssd': { label: 'ZNSSD 上限', help: 'ZNSSD = 2(1 − ZNCC)' },
  'dic.stereo.max_reprojection_px': { label: '重投影误差上限', unit: 'px' },
  'dic.stereo.seed_radius_px': {
    label: 'SIFT 种子半径',
    unit: 'px',
    advanced: true,
  },
  'dic.temporal': { label: '时序匹配 L0 → Lt' },
  'dic.temporal.shape_order': { label: '形函数阶数' },
  'dic.temporal.initialization': {
    label: '初值',
    help: 'previous_solution：上一帧的解作为初值（推荐）',
  },
  'dic.temporal.frame_batch': {
    label: '写盘批大小',
    unit: '帧',
    advanced: true,
  },
  'dic.temporal.max_znssd': { label: 'ZNSSD 上限' },
  'dic.temporal.max_jump_px': {
    label: '帧间跳变上限',
    unit: 'px',
    help: '留空表示不检查',
  },
  'dic.quality': { label: '质量检查', advanced: true },
  'dic.quality.fb_stride': {
    label: '前后向抽检间隔',
    unit: '帧',
    advanced: true,
  },
  'dic.storage': { label: '存储', advanced: true },
  'dic.storage.save_params': {
    label: '保存逐点形函数参数',
    help: '体积很大，仅调试时打开',
    advanced: true,
  },

  recon: { label: '重建' },
  'recon.span_direction': {
    label: '展向参考方向',
    tuple: ['x', 'y', 'z'],
    help: '左相机坐标系中叶片展向的大致方向，用来确定局部坐标系的 s 轴',
  },
  'recon.normal_radius_steps': {
    label: '法向 PCA 半径',
    unit: '× 网格步长',
    advanced: true,
  },
}

const SIDE = /^(left|right)$/

export function hintFor(path: string[]): Hint {
  const key = path.join('.')
  if (HINTS[key]) return HINTS[key]
  const wild = path.map((p) => (SIDE.test(p) ? '*' : p)).join('.')
  return HINTS[wild] ?? {}
}

export function labelFor(path: string[]): string {
  const last = path[path.length - 1]
  if (last === 'left') return '左相机'
  if (last === 'right') return '右相机'
  return hintFor(path).label ?? last
}
