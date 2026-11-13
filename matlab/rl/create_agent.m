function agent=create_agent(algorithm,seed,device)
if nargin<2,seed=42;end
if nargin<3,device='cpu';end
rng(seed);c=read_training_config();a=c.(algorithm);
obs=rlNumericSpec([14 1]);obs.Name='road_preview_v2';
act=rlNumericSpec([4 1],'LowerLimit',-ones(4,1),'UpperLimit',ones(4,1));act.Name='adaptive_mpc';
actorNet=[featureInputLayer(14,'Normalization','none','Name','state');
 fullyConnectedLayer(a.actor_hidden_width);reluLayer;
 fullyConnectedLayer(a.actor_hidden_width);reluLayer;
 fullyConnectedLayer(4);tanhLayer];
actor=rlContinuousDeterministicActor(dlnetwork(actorNet),obs,act,'ObservationInputNames','state','UseDevice',device);
statePath=[featureInputLayer(14,'Normalization','none','Name','state') fullyConnectedLayer(256,'Name','state_fc')];
actionPath=[featureInputLayer(4,'Normalization','none','Name','action') fullyConnectedLayer(256,'Name','action_fc')];
common=[additionLayer(2,'Name','sum') reluLayer fullyConnectedLayer(256) reluLayer fullyConnectedLayer(1)];
g=layerGraph(statePath);g=addLayers(g,actionPath);g=addLayers(g,common);g=connectLayers(g,'state_fc','sum/in1');g=connectLayers(g,'action_fc','sum/in2');
critic1=rlQValueFunction(dlnetwork(g),obs,act,'ObservationInputNames','state','ActionInputNames','action','UseDevice',device);
if strcmp(algorithm,'ddpg')
    opts=rlDDPGAgentOptions;opts.NoiseOptions=rl.option.GaussianActionNoise;
else
    opts=rlTD3AgentOptions;opts.PolicyUpdateFrequency=a.policy_delay;
    opts.TargetPolicySmoothModel.StandardDeviation=a.target_policy_noise;
    opts.TargetPolicySmoothModel.LowerLimit=-a.target_noise_clip;opts.TargetPolicySmoothModel.UpperLimit=a.target_noise_clip;
end
opts.SampleTime=.08;opts.DiscountFactor=.995967610541;
opts.ExperienceBufferLength=c.training.buffer_size;opts.MiniBatchSize=c.training.batch_size;
opts.NumWarmStartSteps=c.training.learning_starts;opts.LearningFrequency=c.training.train_freq;
opts.NumEpoch=1;opts.MaxMiniBatchPerEpoch=c.training.gradient_steps;
opts.TargetSmoothFactor=a.tau;
opts.ActorOptimizerOptions.LearnRate=a.actor_learning_rate;
for k=1:numel(opts.CriticOptimizerOptions),opts.CriticOptimizerOptions(k).LearnRate=a.critic_learning_rate;end
if strcmp(algorithm,'ddpg')
    opts.NoiseOptions.StandardDeviation=a.exploration_noise_sigma;
    opts.NoiseOptions.StandardDeviationDecayRate=0;
else
    opts.ExplorationModel.StandardDeviation=a.exploration_noise_sigma;
    opts.ExplorationModel.StandardDeviationDecayRate=0;
end
if strcmp(algorithm,'ddpg')
    agent=rlDDPGAgent(actor,critic1,opts);
else
    critic2=rlQValueFunction(dlnetwork(g),obs,act,'ObservationInputNames','state','ActionInputNames','action','UseDevice',device);
    agent=rlTD3Agent(actor,[critic1 critic2],opts);
end
end
