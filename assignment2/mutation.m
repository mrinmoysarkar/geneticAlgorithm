function new_pop = mutation(pop,pm,hash)
l = size(pop,2)-2;
alphabet1=[1,hash];
alphabet2=[0,hash];
alphabet3=[0,1];

for i=1:size(pop,1)
    for j=1:l
        if rand <= pm
            if pop(i,j) == 0
                pop(i,j) = alphabet1(randi(2,1));
            elseif pop(i,j) == 1
                pop(i,j) = alphabet2(randi(2,1));
            else
                pop(i,j) = alphabet3(randi(2,1));
            end
        end
    end
end
new_pop=pop;
end