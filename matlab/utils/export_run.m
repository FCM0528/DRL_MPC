function metrics=export_run(out,label,plotsOnly)
if nargin<3,plotsOnly=false;end
root=fileparts(fileparts(mfilename('fullpath')));folder=fullfile(root,'results',label);
if ~isfolder(folder),mkdir(folder);end
s=out.state_log.Data;t=out.state_log.Time;d=out.control_log.Data;
r=out.reference_log.Data;e=out.error_log.Data;
assert(all(isfinite(s),'all')&&all(isfinite(d),'all'));
metrics=struct('controller',string(label),'lateral_rmse_m',sqrt(mean(e(:,1).^2)), ...
 'max_abs_lateral_m',max(abs(e(:,1))),'heading_rmse_rad',sqrt(mean(e(:,2).^2)), ...
 'qp_failures',sum(d(:,2)==0),'completed',s(end,1)>=149.9, ...
 'end_x_m',s(end,1),'end_time_s',t(end),'samples',numel(t));
if ~plotsOnly
save(fullfile(folder,'raw.mat'),'out','metrics','-v7.3');
T=array2table([t s e d],'VariableNames',{'time_s','X_m','Y_m','Vx_mps','Vy_mps','Yaw_rad','YawRate_radps','Ax_mps2','Ay_mps2','SteerL_rad','SteerR_rad','Beta_rad','RoadCurvature','AlphaL1_rad','AlphaR1_rad','AlphaL2_rad','AlphaR2_rad','FyL1_N','FyR1_N','FyL2_N','FyR2_N','ey_m','epsi_rad','command_rad','qp_converged','iterations','primal','dual','Np','Qy','Qpsi','Rdelta'});
writetable(T,fullfile(folder,'trace.csv'));writetable(struct2table(metrics),fullfile(folder,'metrics.csv'));
end
endStation=min(plot_station_limit(),s(end,1));endTime=interp1(s(:,1),t,endStation);
f=figure('Visible','off','Position',[100 100 1200 850]);tiledlayout(4,2);
nexttile;plot(s(:,1),s(:,2),s(:,1),r(:,1),'--');xlabel('X (m)');ylabel('Y (m)');legend('CarSim','DLC SYX');grid on;
nexttile;plot(t,e);legend('e_y (m)','e_\psi (rad)');grid on;
nexttile;plot(t,rad2deg([d(:,1),s(:,9:10)]));ylabel('Front wheel (deg)');legend('Command','Left','Right');grid on;
nexttile;plot(t,[0;diff(rad2deg(d(:,1)))./diff(t)]);ylabel('Steering rate (deg/s)');grid on;
nexttile;plot(t,rad2deg(s(:,[6 11])));legend('Yaw rate (deg/s)','Sideslip (deg)');grid on;
nexttile;stairs(t,d(:,6));ylabel('Np');grid on;
nexttile;stairs(t,d(:,7:9));legend('Qy','Qpsi','Rdelta');grid on;
nexttile;plot(t,d(:,2:3));legend('QP converged','Iterations');grid on;
axesList=findall(f,'Type','axes');
for ax=axesList'
    if strcmp(ax.XLabel.String,'X (m)'),xlim(ax,[0 endStation]);else,xlim(ax,[0 max(eps,endTime)]);end
end
sgtitle(strrep(label,'_',' '));savefig(f,fullfile(folder,'tracking.fig'));exportgraphics(f,fullfile(folder,'tracking.png'),'Resolution',160);close(f);
disp(struct2table(metrics));
end
