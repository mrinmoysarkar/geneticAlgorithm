function new_pop = reproduction(pop,fit)
f = fit(pop);
f = f/sum(f);
f = cumsum(f);
m = size(pop,1);
new_pop = zeros(size(pop));
for i=1:m
    tem = find((rand<=f)==1);
    new_pop(i,:) = pop(tem(1),:);
end