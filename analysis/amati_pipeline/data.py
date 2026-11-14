from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LN10 = np.log(10.)
MU_CONST = -5*np.log10(3.0856775814913673e24)+25


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def json_write(path, value):
    def native(obj):
        if isinstance(obj,np.generic):return obj.item()
        if isinstance(obj,np.ndarray):return obj.tolist()
        if isinstance(obj,Path):return str(obj)
        raise TypeError(f'Unsupported JSON object: {type(obj)}')
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,default=native),encoding='utf-8')


@dataclass
class Dataset:
    grb:pd.DataFrame
    sn:pd.DataFrame
    hii:pd.DataFrame
    sn_cov:np.ndarray
    source:Path

    @classmethod
    def load(cls,path=None):
        p=Path(path or ROOT/'data').resolve()
        if (p/'manifest.json').exists():
            frozen=json.loads((p/'manifest.json').read_text(encoding='utf-8'))
            expected_hashes=frozen.get('generated_hashes',{})
            if frozen.get('algorithm')=='milp' and 'files' in frozen:
                expected_hashes={**frozen['files'],**expected_hashes}
            for name,expected in expected_hashes.items():
                if sha256(p/name)!=expected:raise ValueError(f'Frozen dataset hash mismatch: {p/name}')
        data=cls(*(pd.read_csv(p/f'{name}.csv',dtype={'grb_id':str,'anchor_id':str,'calibrator_id':str})
                   for name in ['grb','sn','hii']),np.load(p/'sn_cov.npy'),p)
        data.validate()
        return data

    def validate(self):
        if not self.grb.grb_id.is_unique:raise ValueError('GRB IDs must be unique')
        ids=set(self.grb.grb_id)
        for table in [self.sn,self.hii]:
            if not table.grb_id.is_unique:raise ValueError('Each family must use each GRB once')
            if not set(table.grb_id)<=ids:raise ValueError('Unknown GRB ID')
        if self.sn_cov.shape!=(len(self.sn),len(self.sn)):raise ValueError('SN covariance shape mismatch')
        if not np.allclose(self.sn_cov,self.sn_cov.T,atol=1e-12):raise ValueError('Asymmetric SN covariance')
        if len(self.sn):np.linalg.cholesky(self.sn_cov)
        # Optional provenance fields (e.g. equivalent widths) can be unavailable
        # in older surveys. Every quantity used by the likelihood must be finite.
        required=[(self.grb,['z','ep_keV','ep_err_keV','fluence','fluence_err']),
                  (self.sn,['calibrator_z','m_b_corr']),
                  (self.hii,['calibrator_z','log_sigma','log_flux','err_log_sigma','err_log_flux'])]
        for table,columns in required:
            if not np.isfinite(table[columns].to_numpy(dtype=float)).all():raise ValueError('Nonfinite input')
        if 'physical_id' in self.hii and not self.hii.physical_id.is_unique:
            raise ValueError('Repeated physical HII calibrator')
        if np.any(self.grb[['ep_keV','ep_err_keV','fluence','fluence_err','z']].to_numpy()<=0):
            raise ValueError('GRB energies, fluences, errors and redshifts must be positive')
        if np.any(self.hii[['err_log_sigma','err_log_flux']].to_numpy()<0):raise ValueError('Negative HII uncertainty')
