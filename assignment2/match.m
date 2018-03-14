function p=match(msg,pop,hash,Cbid,Ctax,Cext,C,R,ite,Pc,Pm)
n=size(pop,1);
y = ones(n,1);
Ebid = zeros(n,1);
k=1;
for j=1:n
    classifier = pop(j,:);
    for i=1:length(msg)-1
        if msg(i) ~= classifier(i) && classifier(i) ~= hash
            y(j) = 0;%no match
            break;
        end
    end
    if y(j)==1 && msg(length(msg)) == classifier(length(msg))
        y(j) = 2;%winner
        gapop(k,:) = classifier;
        k=k+1;
    end
    if y(j)==2
        Ebid(j) = pop(j,end)*Cbid + randn;
    end
end
maxEbid = max(Ebid);

for j=1:n
    if y(j) == 0
        pop(j,end) = pop(j,end)*(1-Cext);
    elseif y(j)==1
        pop(j,end) = pop(j,end)*(1-Ctax-Cext);
    elseif y(j)==2
        if maxEbid == Ebid(j)
            pop(j,end) = pop(j,end)*(1-Cbid-Ctax-Cext) + R*(1+C*noOfHash(pop(j,:),hash)/(length(pop(j,:))-2));
        else
            pop(j,end) = pop(j,end)*(1-Ctax-Cext);
        end
    end
end
if mod(ite,10) == 0 && k>2
    if k>3
        new_pop = reproduction(gapop);
    else
        new_pop = gapop;
    end
    newpop = cross(new_pop,Pc);
    newpop = mutation(newpop,Pm,hash,msg);
    pop(n+1:n+2,:) = newpop;
    [mn, mni] = min(pop(:,end));
    pop(mni,:) = [];
    [mn, mni] = min(pop(:,end));
    pop(mni,:) = [];
end
p=pop;
end