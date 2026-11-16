import copy
import pytest
from report import unique_validation_rows


def records():
    template=dict(case='0',seed='42',rmse_y_m='.05',rmse_heading_deg='.6',rmse_course_deg='.5',
        peak_y_m='.15',peak_heading_deg='2',distance_m='120.01',qp_failures='0',
        terminated='True',truncated='False',failed='False',legal='True',experiment_signature='real-case-fixture')
    rows=[dict(template,controller=c) for c in ('original_mpc','strong_mpc','ddpg','td3','tqc')]
    return rows


def test_identical_paired_baselines_deduplicate_without_changing_policies():
    rows=records();rows.insert(3,dict(rows[0],mpc_solve_mean_ms='.25'))
    result=unique_validation_rows(rows)
    assert [r['controller'] for r in result]==['original_mpc','strong_mpc','ddpg','td3','tqc']
    assert 'mpc_solve_mean_ms' not in result[0]  # First repeat, never select a faster run.


@pytest.mark.parametrize('field,value',[('rmse_y_m','.06'),('legal','False'),('experiment_signature','other')])
def test_inconsistent_baseline_repetitions_cannot_be_hidden(field,value):
    rows=records();rows.append(dict(rows[0],**{field:value}))
    with pytest.raises(ValueError,match='baseline'):unique_validation_rows(rows)


def test_multiple_policies_or_seeds_require_explicit_reporting():
    rows=records();rows.append(copy.deepcopy(rows[-1]))
    with pytest.raises(ValueError,match='Duplicate policy'):unique_validation_rows(rows)
    rows[-1]['seed']='43'
    with pytest.raises(ValueError,match='seed'):unique_validation_rows(rows)
