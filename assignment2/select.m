function [p1,p2]=select(pop)
f = pop(:,end);
f = f/sum(f);
f = cumsum(f);
tem = find((rand<=f)==1);
p1 = tem(1);
while 1
    tem = find((rand<=f)==1);
    p2 = tem(1);
    if p1~=p2 
        break;
    end
end
end