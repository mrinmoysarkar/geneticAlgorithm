function new_pop = mutation(pop,pm)
m = size(pop,1);
l = size(pop,2);
new_pop = pop;
for i=1:m
    for j=1:l
        if rand <= pm
            new_pop(i,j) = xor(new_pop(i,j),1);
        end
    end
end
end