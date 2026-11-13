function fix_local_spacing()
% Selective fixes for the two close port pairs found by the spacing audit.
model='DRL_MPC';scope=migration_id('CarSim_Vehicle_Plant');
configure(scope,{'Raw_CarSim_Channels','Out1'},[245 135 365 165;285 55 315 75]);
scope=migration_id('Observation_Builder',migration_id('DRL_Adaptive_Tuning'));
configure(scope,{'In1','In2','Road_Preview_V2','Out1'}, ...
    [20 180 50 194;20 90 50 104;130 90 230 190;310 130 340 150]);
    function configure(parent,names,positions)
        ops=cell(1,numel(names));
        for k=1:numel(names)
            id=migration_id(names{k},parent);
            ops{k}=struct('op','configure','target',id,'params', ...
                struct('Position',mat2str(positions(k,:)),'FontName','宋体'));
        end
        result=model_edit(model,parent,jsonencode(ops),'incremental');
        assert(~contains(result,'status: partial'),result);
        check=model_check(model,parent);assert(contains(check,'status: healthy'),check);
    end
end
