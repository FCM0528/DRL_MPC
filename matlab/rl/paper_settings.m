function c=paper_settings()
% Paper Tables 1-3; explicit adaptations for this CarSim vehicle.
c.ts=.04;c.np=40;c.nc=12;c.episodes=300;c.seed=42;
c.qy=[25 55 85 115 155 200 240];c.qpsi=[100 300 500 700 900 1000 1100 1300];
c.lambda=[-1 -.7 -.5 -.3 0 .3 .5 1];
c.qy_gain=80;c.qpsi_gain=4;c.rdelta=13500;
c.scales=[.25;.1;2;.5;30;.05]; % Not specified by paper; SI engineering choices.
c.reward_weights=[300;50;60;15;15;20];
c.gamma=.99;c.actor_lr=1e-4;c.critic_lr=8e-4;c.tau=.01;c.target_frequency=10;
c.noise=.4;c.batch=256;c.buffer=100000;
end
