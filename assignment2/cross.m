function new_pop = cross(pop,pc,p1,p2)
l = size(pop,2)-2;
k = randi(l-1,1,1);
new_pop = pop([p1,p2],:);
if rand <= pc
    new_pop(1,:) = [pop(p1,1:k) pop(p2,k+1:end)];
    new_pop(2,:) = [pop(p2,1:k) pop(p1,k+1:end)];
end
new_pop(1,end) = (1/3)*(pop(p1,end)+pop(p2,end));
new_pop(2,end) = (1/3)*(pop(p1,end)+pop(p2,end));
end