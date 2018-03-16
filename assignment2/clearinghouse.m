function [y,correct] = clearinghouse(pop,m,Cext,Ctax,Cbid,R,msg,hash)
correct = 0;
n = size(pop,1);
Ebid = zeros(n,1);
for i=1:n
    if m(i) == 1
        Ebid(i) = pop(i,end)*Cbid + randn - 0.5;
    end
end
[ma, mai]=max(Ebid);
for i=1:n
    if (i==mai) %|| (sum(pop(i,1:end-1)==pop(mai,1:end-1)) == length(pop(mai,1:end-1)))
        if pop(i,length(msg)) == msg(end)
            pop(i,end) = (1-Cext-Ctax-Cbid)*pop(i,end)+R*(1+8*noOfHash(pop(i,:),hash)/6);
            correct = 1;
        else
            pop(i,end) = (1-Cext-Ctax-Cbid)*pop(i,end);
        end
    elseif m(i) == 1
        pop(i,end) = (1-Cext-Ctax)*pop(i,end);
    else
        pop(i,end) = (1-Cext)*pop(i,end);
    end
end
y = pop;
end