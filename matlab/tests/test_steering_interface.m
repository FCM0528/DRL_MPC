% Real CarSim/Simulink authority test; commands are road-wheel degrees.
root = fileparts(fileparts(mfilename('fullpath')));
setup();
commands = [0 1 -1];
report = zeros(3,7);
for j = 1:3
    in = Simulink.SimulationInput('DRL_MPC');
    in = in.setVariable('steering_test_deg',commands(j));
    in = in.setVariable('steering_interface_test',1);
    duration=1;if commands(j)==0,duration=4;end
    in = in.setModelParameter('StopTime',num2str(duration));
    out = sim(in);
    raw = out.state_log;
    assert(size(raw.Data,2)==20 && all(isfinite(raw.Data),'all'));
    tail = raw.Data(raw.Time>duration-.2,:);
    if commands(j)==0
        assert(max(abs(out.reference_log.Data(:,1)))>3,'Zero test must cross the DLC steering region');
        assert(max(abs(rad2deg(raw.Data(:,9:10))),[],'all')<.1,'Driver steering affected actual wheels during DLC');
    end
    report(j,:) = [commands(j),rad2deg(mean(tail(:,9:10))),rad2deg(mean(tail(:,6))),raw.Data(end,1:2),raw.Time(end)];
    native=out.raw_carsim;
    save(fullfile(root,'results',sprintf('steering_%+d.mat',commands(j))),'raw','native');
end
T = array2table(report,'VariableNames',{'command_deg','left_deg','right_deg','yaw_rate_deg_s','end_x_m','end_y_m','end_time_s'});
writetable(T,fullfile(root,'results','steering_authority.csv'));
disp(T);
assert(all(report(2,2:3)>0.5) && all(report(3,2:3)<-0.5),'External wheel steering authority failed');
assert(all(abs(report(1,2:3))<0.1),'Internal driver overrides zero command');
save_system('DRL_MPC');
