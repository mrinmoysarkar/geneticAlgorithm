function new_pop = cross(pop,pc)
n = floor(size(pop,1)/2);
l = size(pop,2)-2;
k = randi(l-1,1,1);
new_pop = pop;
if rand <= pc
    new_pop(1,:) = [pop(1,1:k) pop(2,k+1:end)];
    new_pop(2,:) = [pop(2,1:k) pop(1,k+1:end)];
    new_pop(1,end) = (1/3)*(pop(1,end)+pop(2,end));
    new_pop(2,end) = (1/3)*(pop(1,end)+pop(2,end));
end
end