function value=matlabPlanValue(x,plan)
%#codegen
if x<=plan(1,1),value=plan(1,2);return;end
if x>=plan(end,1),value=plan(end,2);return;end
index=min(size(plan,1)-1,max(1,floor((x-plan(1,1))/(plan(2,1)-plan(1,1)))+1));
fraction=(x-plan(index,1))/(plan(index+1,1)-plan(index,1));value=plan(index,2)+fraction*(plan(index+1,2)-plan(index,2));
end
