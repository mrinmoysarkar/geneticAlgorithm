function new_pop = mutation(pop,pm,hash,msg)
m = size(pop,1);
l = size(pop,2)-2;
new_pop = pop;
for i=1:2
    for j=1:l
        if rand <= pm
            if new_pop(i,j) == hash
                new_pop(i,j) = msg(j);
            else
                new_pop(i,j) = hash;
            end
        end
    end
end
end