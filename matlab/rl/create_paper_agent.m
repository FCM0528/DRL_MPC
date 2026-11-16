function agent=create_paper_agent(seed)
if nargin<1,seed=42;end
rng(seed);c=paper_settings();obs=rlNumericSpec([6 1]);act=rlNumericSpec([3 1],'LowerLimit',-ones(3,1),'UpperLimit',ones(3,1));
actor=rlContinuousDeterministicActor(dlnetwork([featureInputLayer(6,'Normalization','none');fullyConnectedLayer(256);reluLayer;fullyConnectedLayer(256);reluLayer;fullyConnectedLayer(3);tanhLayer]),obs,act);
g=layerGraph([featureInputLayer(6,'Normalization','none','Name','s');fullyConnectedLayer(256,'Name','sf')]);
g=addLayers(g,[featureInputLayer(3,'Normalization','none','Name','a');fullyConnectedLayer(256,'Name','af')]);
g=addLayers(g,[additionLayer(2,'Name','sum');reluLayer;fullyConnectedLayer(256);reluLayer;fullyConnectedLayer(1)]);
g=connectLayers(g,'sf','sum/in1');g=connectLayers(g,'af','sum/in2');
critics=[rlQValueFunction(dlnetwork(g),obs,act,'ObservationInputNames','s','ActionInputNames','a'),rlQValueFunction(dlnetwork(g),obs,act,'ObservationInputNames','s','ActionInputNames','a')];
o=rlTD3AgentOptions;o.SampleTime=c.ts;o.DiscountFactor=c.gamma;o.TargetSmoothFactor=c.tau;o.TargetUpdateFrequency=c.target_frequency;
o.PolicyUpdateFrequency=2;o.MiniBatchSize=c.batch;o.ExperienceBufferLength=c.buffer;o.NumWarmStartSteps=256;
o.LearningFrequency=1;o.NumEpoch=1;o.MaxMiniBatchPerEpoch=1;
o.ActorOptimizerOptions.LearnRate=c.actor_lr;
for k=1:2,o.CriticOptimizerOptions(k).LearnRate=c.critic_lr;end
o.ExplorationModel.StandardDeviation=c.noise;o.ExplorationModel.StandardDeviationMin=.05;o.ExplorationModel.StandardDeviationDecayRate=5e-5;
agent=rlTD3Agent(actor,critics,o);
end
