function new_pop = reproduction(pop)
for i=1:2
    m = size(pop,1);
    n = size(pop,2);
    f = pop(:,n);
    f = f/sum(f);
    f = cumsum(f);
    tem = find((rand<=f)==1);
    new_pop(i,:) = pop(tem(1),:);
    pop(tem(1),:)=[];
end
end