function new_pop = cross(pop,pc)
n = floor(size(pop,1)/2);
l = size(pop,2);
new_pop = pop;
for i=1:n
    k = randi(l-1,1,1);
    m = size(pop,1);
    mm = 1:m;
    i1 = randi(m,1,1);
    mm(i1)=[];
    i2 = mm(randi(length(mm),1,1));
    if rand <= pc
        new_pop(i1,:) = [pop(i1,1:k) pop(i2,k+1:l)];
        new_pop(i2,:) = [pop(i2,1:k) pop(i1,k+1:l)];
    end
end
end