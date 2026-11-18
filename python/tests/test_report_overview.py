"""The checked-in offline example must preserve the real validation evidence."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def test_all_validation_cases_and_baselines_preserve_saved_metrics():
    old=list(csv.DictReader((ROOT/'datasets/carsim/reports/dataset_v1_seed42_pilot/pilot_validation_metrics.csv').open(encoding='utf-8-sig')))
    new=list(csv.DictReader((ROOT/'reports/research_overview_20261118/metrics.csv').open(encoding='utf-8-sig')))
    assert {float(r['speed_kmh']) for r in new}=={65.,75.}
    assert sum(r['kind']=='baseline' for r in new)==4
    for row in old:
        actual=next(r for r in new if all(r[k]==row[k] for k in ('algorithm','seed','kind','speed_kmh')))
        for key in ('rmse_y_m','rmse_heading_deg','peak_y_m','peak_heading_deg','strong_iy','strong_ih'):
            np.testing.assert_allclose(float(actual[key]),float(row[key]),rtol=0,atol=1e-12)


def test_single_seed_has_no_fabricated_uncertainty_and_fixed_np_is_not_adaptive():
    report=ROOT/'reports/research_overview_20261118'
    summary=json.loads((report/'summary.json').read_text(encoding='utf-8'))
    assert len(summary['cases'])==2
    assert all(r['n_seeds']==1 and r['rmse_y_m_sample_sd'] is None for r in summary['seed_aggregates'])
    assert all('np' not in r['policy_action_names'] for r in summary['configurations'])
    page=(report/'index.html').read_text(encoding='utf-8')
    assert 'figures/executed_weights.png' in page and 'summary.json' in page
    assert 'fixed' in (report/'metrics.csv').read_text(encoding='utf-8-sig')
