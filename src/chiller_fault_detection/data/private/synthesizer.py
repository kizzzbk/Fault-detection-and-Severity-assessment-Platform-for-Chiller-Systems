from pathlib import Path

import CoolProp.CoolProp as CP
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from chiller_fault_detection.config import PROCESSED_DIR
from chiller_fault_detection.data.public.load import FEATURE_COLUMNS, METADATA_COLUMNS
from chiller_fault_detection.labels import LABEL_TO_ID, FaultType

FLUID = 'R134a'
PSI_PER_PA = 6894.757
ATM_PSI = 14.696
ETA_MOTOR = 0.92
ETA_IS_ANCHOR = 0.98
DEFAULT_SUBCOOLING_F = 5.0
DEFAULT_ANCHOR = PROCESSED_DIR / 'laguna_processed.parquet'
DEFAULT_SYNTHETIC = PROCESSED_DIR / 'laguna_synthetic_normal.parquet'
OUTPUT_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS


def as_preview_table(table):
  """string thường, không metadata pandas, để trình xem Parquet đọc được."""
  fields = [
      field.with_type(pa.string()) if pa.types.is_large_string(field.type) else field
      for field in table.schema
  ]
  return table.cast(pa.schema(fields))


def open_parquet_writer(path, schema):
  """Parquet 1.0, mã hóa PLAIN. Bản 2.6 kèm dictionary trên cột double làm Kaggle preview hỏng."""
  return pq.ParquetWriter(
      path,
      schema,
      compression='snappy',
      version='1.0',
      use_dictionary=False,
      data_page_version='1.0',
      store_schema=False,
  )


def write_preview_table(writer, table, split_first=False, first_group_rows=1000):
  """Nhóm dòng đầu ngắn để trang preview không phải trả cả 100.000 dòng dưới dạng JSON."""
  if split_first and table.num_rows > first_group_rows:
    writer.write_table(table.slice(0, first_group_rows))
    writer.write_table(table.slice(first_group_rows))
    return
  writer.write_table(table)


CLOSURE_ITERS = 3
# Severity 1–4 as a fraction of the RP-1043 test. Two filenames share a level
# for CF (6% and 12%) and EO (68% and 73%); the primary test is kept.
RP1043_FRACTION = {
    FaultType.FWE: (0.10, 0.20, 0.30, 0.40),
    FaultType.FWC: (0.10, 0.20, 0.30, 0.40),
    FaultType.CF: (0.12, 0.20, 0.30, 0.45),
    FaultType.RO: (0.10, 0.20, 0.30, 0.40),
    FaultType.RL: (0.10, 0.20, 0.30, 0.40),
    FaultType.NC: (0.01, 0.02, 0.03, 0.05),
    FaultType.EO: (0.14, 0.32, 0.50, 0.68),
}


def _f_to_k(t_f):
  return (t_f - 32.0) * 5.0 / 9.0 + 273.15


def _k_to_f(t_k):
  return (t_k - 273.15) * 9.0 / 5.0 + 32.0


def _psig_to_pa(p_psig):
  return (p_psig + ATM_PSI) * PSI_PER_PA


def _pa_to_psig(p_pa):
  return (p_pa / PSI_PER_PA) - ATM_PSI


def _saturation_psig(t_f):
  return _pa_to_psig(CP.PropsSI('P', 'T', _f_to_k(t_f), 'Q', 1, FLUID))


def _cycle_capacities(tre_f, trc_f, kw, tsh_f, sub_f, eta_is, eta_motor):
  """Qe and Qc at one measured state. Suction pressure is Psat(TRE)."""
  p_suc = _psig_to_pa(_saturation_psig(tre_f))
  p_dis = _psig_to_pa(_saturation_psig(trc_f))
  h_suc = CP.PropsSI('H', 'T', _f_to_k(tre_f + tsh_f), 'P', p_suc, FLUID)
  s_suc = CP.PropsSI('S', 'T', _f_to_k(tre_f + tsh_f), 'P', p_suc, FLUID)
  h_dis_is = CP.PropsSI('H', 'P', p_dis, 'S', s_suc, FLUID)
  h_dis = h_suc + (h_dis_is - h_suc) / eta_is
  h_liq = CP.PropsSI('H', 'T', _f_to_k(trc_f - sub_f), 'P', p_dis, FLUID)
  m_ref = eta_motor * kw * 1000.0 / (h_dis - h_suc)
  qe = m_ref * (h_suc - h_liq) / 1000.0
  qc = qe + eta_motor * kw
  return float(qe), float(qc)


class CalibratedLagunaChiller:
  """Bộ mô phỏng Chiller Laguna giải theo chu trình CoolProp và neo dữ liệu thực nghiệm."""

  def __init__(self, anchor_path=DEFAULT_ANCHOR):
    frame = pd.read_parquet(anchor_path)
    if frame.empty:
      raise ValueError(f'Anchor parquet has no rows: {anchor_path}')
    med = frame.median(numeric_only=True)
    sub = frame['TRC_sub'].median()

    self.anchor_path = anchor_path
    self.kW0 = float(med['kW'])
    self.TEO_setpoint = float(med['TEO'])
    self.tsh0 = float(med['Tsh_suc'])
    self.trc_sub0 = DEFAULT_SUBCOOLING_F if pd.isna(sub) else float(sub)
    self.chw_dt0 = float(med['CHW_DT'])
    self.cdw_dt0 = float(med['CDW_DT'])
    self.pinch0 = float(med['TEO'] - med['TRE'])
    self.tca0 = float(med['TRC'] - med['TCO'])
    self.pre_bias = float(med['PRE'] - _saturation_psig(med['TRE']))
    self.po_net0 = float(med['PO_net'])
    self.eta_motor = ETA_MOTOR
    self.eta_is0 = ETA_IS_ANCHOR
    self.Qe0, self.Qc0 = _cycle_capacities(
        float(med['TRE']),
        float(med['TRC']),
        self.kW0,
        self.tsh0,
        self.trc_sub0,
        self.eta_is0,
        self.eta_motor,
    )

  def solve_single_step(
      self, kw, tci_f, teo_f=None, add_noise=True, fault='normal', severity=0
  ):
    """Một điểm vận hành. Cùng solver với ``solve_batch``."""
    frame = solve_batch(
        self,
        [kw],
        [tci_f],
        self.TEO_setpoint if teo_f is None else teo_f,
        np.random.default_rng(),
        add_noise=add_noise,
        fault=fault,
        severity=severity,
    )
    return frame.iloc[0].to_dict()


def load_profile(total_samples, step_sec=10.0, seed=42):
  """Hold 20–60 phút và ramp 6–18 phút, TCI dao động 24 giờ quanh 81°F."""
  rng = np.random.default_rng(seed)
  time_arr = np.arange(total_samples, dtype=np.float64) * step_sec
  tci_profile = 81.0 + 6.0 * np.sin(2 * np.pi * time_arr / 86400.0 - np.pi / 2)
  kw_profile = np.empty(total_samples, dtype=np.float64)
  idx = 0
  current_kw = 187.0
  while idx < total_samples:
    hold_len = int(rng.integers(120, 360))
    end_hold = min(idx + hold_len, total_samples)
    kw_profile[idx:end_hold] = current_kw
    idx = end_hold
    if idx >= total_samples:
      break
    next_kw = float(rng.uniform(90.0, 195.0))
    ramp_len = int(rng.integers(36, 108))
    end_ramp = min(idx + ramp_len, total_samples)
    kw_profile[idx:end_ramp] = np.linspace(current_kw, next_kw, end_ramp - idx)
    current_kw = next_kw
    idx = end_ramp
  return time_arr, kw_profile, tci_profile


def _fault_key(fault):
  key = str(fault).strip().lower()
  if key == 'ca':
    raise ValueError("Fault name 'ca' is not in the training taxonomy; use 'cf'")
  try:
    return FaultType(key)
  except ValueError as exc:
    known = ', '.join(fault.value for fault in FaultType)
    raise ValueError(f'Unknown fault {fault!r}. Expected one of: {known}') from exc


def _rp_fraction(kind, level):
  if level == 0 or kind is FaultType.NORMAL:
    return 0.0
  fraction = RP1043_FRACTION[kind][level - 1]
  if not 0.0 <= fraction < 1.0:
    raise ValueError(f'RP-1043 fraction for {kind.value} SL{level} is {fraction}')
  return fraction


def _fault_offsets(chiller, fault, severity, n):
  """RP-1043 fractions applied with the local balance laws. Anchor is not mutated."""
  kind = _fault_key(fault)
  level = int(severity)
  if level not in range(5):
    raise ValueError(f'severity_level must be 0–4, got {severity}')
  if kind is FaultType.NORMAL:
    level = 0
  ratio = _rp_fraction(kind, level)
  area_mult = 1.0 / (1.0 - ratio) if ratio else 1.0
  tsh = np.full(n, chiller.tsh0)
  sub = np.full(n, chiller.trc_sub0)
  pinch_mult = 1.0
  po_mult = 1.0
  air = 0.0
  chw_mult = cdw_mult = tca_mult = 1.0
  if kind is FaultType.FWE:
    # m_fault = m0 (1-r) so ΔT = ΔT0 / (1-r) at the same Q.
    chw_mult = area_mult
  elif kind is FaultType.FWC:
    cdw_mult = area_mult
  elif kind is FaultType.CF:
    # Q ≈ UA · TCA, so the same Q at a smaller UA raises approach by 1/(1-r).
    tca_mult = area_mult
  elif kind is FaultType.RO:
    sub = chiller.trc_sub0 * (1.0 + 2.0 * ratio)
  elif kind is FaultType.RL:
    sub = np.maximum(0.0, chiller.trc_sub0 * (1.0 - 2.5 * ratio))
    tsh = chiller.tsh0 * (1.0 + 0.8 * ratio)
  elif kind is FaultType.NC:
    air = ratio
  elif kind is FaultType.EO:
    pinch_mult = 1.0 + 0.5 * ratio
    po_mult = 1.0 + 0.3 * ratio
  return {
      'fault': kind.value,
      'fault_id': LABEL_TO_ID[kind],
      'severity': level,
      'ratio': ratio,
      'chw_mult': chw_mult,
      'cdw_mult': cdw_mult,
      'tca_mult': tca_mult,
      'pinch_mult': pinch_mult,
      'air': air,
      'po_mult': po_mult,
      'tsh': tsh,
      'sub': sub,
  }


def solve_batch(chiller, kw, tci, teo, rng, add_noise=True, fault='normal', severity=0):
  """Closed cycle. Subcooling and non-condensable pressure enter before discharge."""
  kw = np.asarray(kw, dtype=np.float64)
  tci = np.asarray(tci, dtype=np.float64)
  n = kw.shape[0]
  if np.isscalar(teo):
    teo = np.full(n, float(teo), dtype=np.float64)
  else:
    teo = np.asarray(teo, dtype=np.float64)

  spec = _fault_offsets(chiller, fault, severity, n)
  tsh = spec['tsh']
  trc_sub = spec['sub']
  if add_noise:
    tsh = tsh + rng.normal(0.0, 0.5, n)
    trc_sub = np.maximum(0.0, trc_sub + rng.normal(0.0, 0.3, n))

  eta_is = np.clip(
      chiller.eta_is0 * (1.0 - 0.22 * (1.0 - kw / chiller.kW0) ** 2),
      0.75,
      0.985,
  )
  w_comp = kw * chiller.eta_motor
  qe = kw * (chiller.Qe0 / chiller.kW0)
  qc = qe + w_comp
  tei = tre = tco = tca = trc = None
  p_suc_pa = p_dis_pa = None
  tr_dis = None
  for _ in range(CLOSURE_ITERS):
    tei = teo + chiller.chw_dt0 * (qe / chiller.Qe0) * spec['chw_mult']
    tre = teo - chiller.pinch0 * (qe / chiller.Qe0) * spec['pinch_mult']
    tco = tci + chiller.cdw_dt0 * (qc / chiller.Qc0) * spec['cdw_mult']
    tca = chiller.tca0 * (qc / chiller.Qc0) * spec['tca_mult']
    trc = tco + tca
    p_suc_pa = CP.PropsSI('P', 'T', _f_to_k(tre), 'Q', 1, FLUID)
    p_sat_pa = CP.PropsSI('P', 'T', _f_to_k(trc), 'Q', 0, FLUID)
    p_dis_pa = p_sat_pa / (1.0 - spec['air']) if spec['air'] else p_sat_pa
    t_suc_k = _f_to_k(tre + tsh)
    h_suc = CP.PropsSI('H', 'T', t_suc_k, 'P', p_suc_pa, FLUID)
    s_suc = CP.PropsSI('S', 'T', t_suc_k, 'P', p_suc_pa, FLUID)
    h_dis_is = CP.PropsSI('H', 'P', p_dis_pa, 'S', s_suc, FLUID)
    h_dis = h_suc + (h_dis_is - h_suc) / eta_is
    sub = np.broadcast_to(np.asarray(trc_sub, dtype=np.float64), trc.shape)
    h_sat = CP.PropsSI('H', 'P', p_sat_pa, 'Q', 0, FLUID)
    t_liq = np.where(sub > 0.05, trc - sub, trc - 0.05)
    h_sub = CP.PropsSI('H', 'T', _f_to_k(t_liq), 'P', p_dis_pa, FLUID)
    h_liq = np.where(sub > 0.05, h_sub, h_sat)
    qe = (w_comp * 1000.0 / (h_dis - h_suc)) * (h_suc - h_liq) / 1000.0
    qc = qe + w_comp
  tr_dis = _k_to_f(CP.PropsSI('T', 'P', p_dis_pa, 'H', h_dis, FLUID))

  if add_noise:
    pre = _pa_to_psig(p_suc_pa) + chiller.pre_bias + rng.normal(0.0, 0.08, n)
    prc = _pa_to_psig(p_dis_pa) + rng.normal(0.0, 0.25, n)
    kw_out = kw + rng.normal(0.0, 0.4, n)
    tr_dis = tr_dis + rng.normal(0.0, 0.3, n)
    to_sump = tr_dis + 10.0 + rng.normal(0.0, 0.3, n)
    po_net = chiller.po_net0 * spec['po_mult'] + rng.normal(0.0, 0.15, n)
    tei = tei + rng.normal(0.0, 0.08, n)
    tco = tco + rng.normal(0.0, 0.08, n)
  else:
    pre = _pa_to_psig(p_suc_pa) + chiller.pre_bias
    prc = _pa_to_psig(p_dis_pa)
    kw_out = kw
    to_sump = tr_dis + 10.0
    po_net = np.full(n, chiller.po_net0 * spec['po_mult'])

  frame = pd.DataFrame(
      {
          'source_file': f'synthetic_{spec["fault"]}',
          'time': np.zeros(n, dtype=np.float64),
          'fault': spec['fault'],
          'fault_id': np.int64(spec['fault_id']),
          'severity_level': np.int64(spec['severity']),
          'path_label': f'{spec["fault"]}/{spec["severity"]}',
          'TEI': tei,
          'TEO': teo,
          'TCI': tci,
          'TCO': tco,
          'TRE': tre,
          'TRC': trc,
          'TRC_sub': trc_sub,
          'Tsh_suc': tsh,
          'TR_dis': tr_dis,
          'TCA': tca,
          'kW': kw_out,
          'TO_sump': to_sump,
          'PO_net': po_net,
          'PRE': pre,
          'PRC': prc,
          'P_lift': prc - pre,
          'CDW_DT': tco - tci,
          'CHW_DT': tei - teo,
      }
  )
  frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].round(2)
  return frame.loc[:, OUTPUT_COLUMNS]


def _block_severity(start_idx, total_samples, fault, severity, severities):
  if severities is None:
    return severity
  if len(severities) == 0 or total_samples % len(severities) != 0:
    raise ValueError('total_samples must divide evenly across severities')
  block = total_samples // len(severities)
  return int(severities[min(start_idx // block, len(severities) - 1)])


def generate_normal_chunks(
    total_samples=2_000_000,
    chunk_size=100_000,
    output_file=DEFAULT_SYNTHETIC,
    anchor_path=DEFAULT_ANCHOR,
    seed=42,
    fault='normal',
    severity=0,
    severities=None,
):
  """Sinh kịch bản Ramp/Hold và ghi Parquet theo khối."""
  output_file = Path(output_file)
  chiller = CalibratedLagunaChiller(anchor_path)
  time_arr, kw_profile, tci_profile = load_profile(total_samples, seed=seed)
  output_file.parent.mkdir(parents=True, exist_ok=True)
  rng = np.random.default_rng(seed)
  writer = None
  print(
      f'{fault} severity={severity if severities is None else list(severities)} '
      f'kW0={chiller.kW0:.2f} Qe0={chiller.Qe0:.1f}. '
      f'Ghi {total_samples:,} dòng → {output_file}',
      flush=True,
  )
  try:
    for start_idx in range(0, total_samples, chunk_size):
      end_idx = min(start_idx + chunk_size, total_samples)
      block_severity = _block_severity(
          start_idx, total_samples, fault, severity, severities
      )
      frame = solve_batch(
          chiller,
          kw_profile[start_idx:end_idx],
          tci_profile[start_idx:end_idx],
          chiller.TEO_setpoint,
          rng,
          fault=fault,
          severity=block_severity,
      )
      frame['time'] = time_arr[start_idx:end_idx]
      table = as_preview_table(pa.Table.from_pandas(frame, preserve_index=False))
      if writer is None:
        writer = open_parquet_writer(output_file, table.schema)
      write_preview_table(writer, table, split_first=start_idx == 0)
      print(f'-> Đã ghi {end_idx:,} / {total_samples:,} mẫu', flush=True)
  finally:
    if writer is not None:
      writer.close()
  print(f'Hoàn thành {total_samples:,} dòng → {output_file}')
  return output_file


def generate_fault_chunks(
    fault,
    severity=None,
    total_samples=2_000_000,
    chunk_size=100_000,
    output_file=None,
    anchor_path=DEFAULT_ANCHOR,
    seed=42,
):
  """Một file lỗi. Mặc định SL1–SL4 chia đều, cùng lịch kW và TCI với Normal."""
  kind = _fault_key(fault)
  if kind is FaultType.NORMAL:
    raise ValueError('Use generate_normal_chunks for the normal file')
  if output_file is None:
    output_file = PROCESSED_DIR / f'{kind.value}.parquet'
  if severity is None:
    return generate_normal_chunks(
        total_samples=total_samples,
        chunk_size=chunk_size,
        output_file=output_file,
        anchor_path=anchor_path,
        seed=seed,
        fault=kind.value,
        severities=(1, 2, 3, 4),
    )
  level = int(severity)
  if level not in (1, 2, 3, 4):
    raise ValueError('severity_level must be 1–4')
  return generate_normal_chunks(
      total_samples=total_samples,
      chunk_size=chunk_size,
      output_file=output_file,
      anchor_path=anchor_path,
      seed=seed,
      fault=kind.value,
      severity=level,
  )


if __name__ == '__main__':
  # 1. Chạy thử nghiệm mỏ neo
  chiller = CalibratedLagunaChiller()
  print(
      f'anchor={chiller.anchor_path} kW0={chiller.kW0:.2f} '
      f'Qe0={chiller.Qe0:.1f} Qc0={chiller.Qc0:.1f} '
      f'TEO={chiller.TEO_setpoint:.2f} PRE_bias={chiller.pre_bias:.2f}'
  )
  print('--- KIỂM TRA ĐỘ KHỚP TẠI MỎ NEO (5 MẪU) ---')
  sample_rows = [
      chiller.solve_single_step(
          chiller.kW0, 81.0, chiller.TEO_setpoint, add_noise=False
      )
      for _ in range(5)
  ]
  df_check = pd.DataFrame(sample_rows)
  print(
      df_check[[
          'TEI',
          'TEO',
          'CHW_DT',
          'TCI',
          'TCO',
          'CDW_DT',
          'TRC',
          'PRC',
          'PRE',
          'P_lift',
          'TO_sump',
          'PO_net',
      ]].head(1)
  )

  pinch_before = chiller.pinch0
  print('--- NHÃN VÀ KHÉP VÒNG TẠI MỎ NEO, severity 2 ---')
  for name in ('normal', 'cf', 'fwe', 'fwc', 'ro', 'rl', 'nc', 'eo'):
    level = 0 if name == 'normal' else 2
    row = chiller.solve_single_step(
        chiller.kW0, 82.22, chiller.TEO_setpoint,
        add_noise=False, fault=name, severity=level,
    )
    print(
        f"{row['path_label']:8} id={row['fault_id']} "
        f"CHW={row['CHW_DT']:6.2f} CDW={row['CDW_DT']:5.2f} "
        f"TCA={row['TCA']:5.2f} sub={row['TRC_sub']:5.2f} "
        f"Tsh={row['Tsh_suc']:5.2f} PRC={row['PRC']:7.2f} "
        f"TRdis={row['TR_dis']:6.2f} PO={row['PO_net']:6.2f}"
    )
  if chiller.pinch0 != pinch_before:
    raise RuntimeError('EO mutated the anchor pinch')