function y = noOfCorrectSol(pop,hash)
n=size(pop,1);
y = 0;
for i=1:n
    classifier = pop(i,:);
    sol = 0;
    if classifier(1) == 0 && classifier(2) == 0 && classifier(3)==classifier(7)
        sol = 1;
    elseif classifier(1) == 0 && classifier(2) == 1 && classifier(4)==classifier(7)
        sol = 1;
    elseif classifier(1) == 1 && classifier(2) == 0 && classifier(5)==classifier(7)
        sol = 1;
    elseif classifier(1) == 1 && classifier(2) == 1 && classifier(6)==classifier(7)
        sol = 1;
    elseif classifier(1) == 0 && classifier(2) == hash && classifier(3)==classifier(7) && classifier(4)==classifier(7)
        sol = 1;
    elseif classifier(1) == hash && classifier(2) == 0 && classifier(3)==classifier(7) && classifier(5)==classifier(7)
        sol = 1;
    elseif classifier(1) == 1 && classifier(2) == hash && classifier(5)==classifier(7) && classifier(6)==classifier(7)
        sol = 1;
    elseif classifier(1) == hash && classifier(2) == 1 && classifier(4)==classifier(7) && classifier(6)==classifier(7)
        sol = 1;
    elseif classifier(1) == hash && classifier(2) == hash && classifier(3)==classifier(7) && classifier(4)==classifier(7) && classifier(5)==classifier(7) && classifier(6)==classifier(7)
        sol = 1;
    end
    y = y+sol;
end
end