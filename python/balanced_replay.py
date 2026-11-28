"""SB3 replay with equal case exposure and native per-transition provenance."""
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer


class CaseBalancedReplayBuffer(ReplayBuffer):
    """Append-only during a locked run. Abort before a circular write can relabel a sample."""

    def __init__(self,*args,scenario_ids,seed,**kwargs):
        super().__init__(*args,**kwargs)
        if self.n_envs!=1 or len(scenario_ids)!=6 or len(set(scenario_ids))!=6:
            raise ValueError('Balanced replay expects six unique cases and one native solver')
        self.scenario_ids=tuple(scenario_ids)
        self.cases=np.full(self.buffer_size,-1,dtype=np.int8)
        self.segments=np.full(self.buffer_size,-1,dtype=np.int8)
        self.by_case=[[] for _ in scenario_ids]
        self.rng=np.random.default_rng(seed)
        self.draws=np.zeros(len(scenario_ids),dtype=np.int64)
        self.balanced_draws=np.zeros(len(scenario_ids),dtype=np.int64)
        self.batches=0
        self.warmup_batches=0

    def add(self,obs,next_obs,action,reward,done,infos):
        if self.full:
            raise RuntimeError('Case-balanced replay is append-only within its declared budget')
        case=self.scenario_ids.index(infos[0]['scenario_id'])
        segment=infos[0]['segment']
        if segment not in (0,1,2):
            raise ValueError('Unknown native wave segment')
        index=self.pos
        super().add(obs,next_obs,action,reward,done,infos)
        self.cases[index]=case
        self.segments[index]=segment
        self.by_case[case].append(index)

    def sample(self,batch_size,env=None):
        eligible=np.flatnonzero([bool(group) for group in self.by_case])
        if not len(eligible):
            raise RuntimeError('No native transitions to sample')
        counts=np.full(len(eligible),batch_size//len(eligible))
        counts[self.rng.permutation(len(eligible))[:batch_size%len(eligible)]]+=1
        indices=np.concatenate([self.rng.choice(self.by_case[case],size=count) for case,count in zip(eligible,counts,strict=True)])
        self.draws[eligible]+=counts
        self.batches+=1
        if len(eligible)==len(self.scenario_ids):
            self.balanced_draws[eligible]+=counts
        else:
            self.warmup_batches+=1
        return self._get_samples(self.rng.permutation(indices),env=env)

    def sampling_audit(self):
        return dict(batches=self.batches,warmup_batches=self.warmup_batches,
                    draws=dict(zip(self.scenario_ids,self.draws.tolist(),strict=True)),
                    balanced_draws=dict(zip(self.scenario_ids,self.balanced_draws.tolist(),strict=True)),
                    transitions_by_segment={name:np.bincount(self.segments[group],minlength=3).tolist()
                                            for name,group in zip(self.scenario_ids,self.by_case,strict=True)})
