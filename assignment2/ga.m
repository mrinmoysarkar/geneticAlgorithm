function new_pop = ga(pop,pc,pm,hash)
new_pop = pop;
[p1,p2] = select(pop);
child = cross(pop,pc,p1,p2);
child = mutation(child,pm,hash);
ff = new_pop(:,end);
[mm,i1] = min(ff);
ff(i1) = ff(i1)+max(ff);
[mm,i2] = min(ff);
new_pop(i1,:)=child(1,:);
new_pop(i2,:)=child(2,:);
new_pop(i1,end)=(2/3)*new_pop(p1,end);
new_pop(i2,end)=(2/3)*new_pop(p2,end);
end