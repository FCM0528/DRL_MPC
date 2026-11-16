from joint_validation import aggregate_cases


def test_one_bad_case_cannot_pass_strong_baseline():
    good=dict(legal=True,original_dual=True,strong_dual=True,strong_iy=.1,strong_ih=.2,
              rmse_y_m=.1,rmse_heading_deg=1)
    bad=dict(good,strong_dual=False,strong_iy=-.01)
    result=aggregate_cases([good,bad],'model.zip',False)
    assert result['original_dual']
    assert not result['strong_dual']
    assert result['score']==-.01
    assert not aggregate_cases([good,dict(bad,legal=False)],'model.zip',False)['legal']
