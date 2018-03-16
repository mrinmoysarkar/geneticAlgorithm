function new_pop = ga(pop,pc,pm,hash)
n = size(pop,1);
new_pop = pop;
f = pop(:,end);
f = f/sum(f);
f = cumsum(f);
tem_pop = zeros(2,size(pop,2));
k=1;
tindx=zeros(2,1);

tem = find((rand<=f)==1);
i1 = tem(1);
parent1 = pop(i1,:);
while 1
    tem = find((rand<=f)==1);
    i2 = tem(1);
    parent2 = pop(i2,:);
    if i1~=i2 % && parent1(end-1) == parent2(end-1)
        break;
    end
end
child(1,:) = parent1;
child(2,:) = parent2;
if rand <= pc
    l = size(pop,2)-2;
    s = randi(l-1,1,1);
    child(1,:) = [parent1(1:s) parent2(s+1:end)];
    child(2,:) = [parent2(1:s) parent1(s+1:end)];
    child(1,end) = (1/3)*(parent1(end)+parent2(end));
    child(2,end) = (1/3)*(parent1(end)+parent2(end));
    new_pop(i1,end)=(2/3)*new_pop(i1,end);
    new_pop(i2,end)=(2/3)*new_pop(i2,end);
end

child = mutation(child,pm,hash);

ff = new_pop(:,end);
[mm,i1] = min(ff);
ff(i1) = ff(i1)+max(ff);
[mm,i2] = min(ff);

new_pop(i1,:)=child(1,:);
new_pop(i2,:)=child(2,:);

% for i=1:2
%     tem = find((rand<=f)==1);
%     tem_pop(k,:) = pop(tem(1),:);
%     tindx(k) = tem(1);
%     k=k+1;
%     if k==3
%         k=1;
%         child1 = tem_pop(1,:);
%         child2 = tem_pop(2,:);
%         if rand <= pc
%             l = size(pop,2)-2;
%             s = randi(l-1,1,1);
%             child1 = [tem_pop(1,1:s) tem_pop(2,s+1:end)];
%             child2 = [tem_pop(2,1:s) tem_pop(1,s+1:end)];
%             child1(end) = (1/3)*(tem_pop(1,end)+tem_pop(2,end));
%             child2(end) = (1/3)*(tem_pop(1,end)+tem_pop(2,end));
%             new_pop(tindx(1),end)=(2/3)*new_pop(tindx(1),end);
%             new_pop(tindx(2),end)=(2/3)*new_pop(tindx(2),end);
%         end
%         for j=1:6
%             if rand <= pm
%                 if child1(j) == 0
%                     if rand<=.5
%                         child1(j) = 1;
%                     else
%                         child1(j) = hash;
%                     end
%                 elseif child1(j) == 1
%                     if rand<=.5
%                         child1(j) = 0;
%                     else
%                         child1(j) = hash;
%                     end
%                 else
%                     if rand<=.5
%                         child1(j) = 0;
%                     else
%                         child1(j) = 1;
%                     end
%                 end
%             end
%         end
%         for j=1:6
%             if rand <= pm
%                 if child2(j) == 0
%                     if rand<=.5
%                         child2(j) = 1;
%                     else
%                         child2(j) = hash;
%                     end
%                 elseif child2(j) == 1
%                     if rand<=.5
%                         child2(j) = 0;
%                     else
%                         child2(j) = hash;
%                     end
%                 else
%                     if rand<=.5
%                         child2(j) = 0;
%                     else
%                         child2(j) = 1;
%                     end
%                 end
%             end
%         end
%         ff = new_pop(:,end);
%         [mm,i1] = min(ff);
%         ff(i1) = ff(i1)+max(ff);
%         [mm,i2] = min(ff);
%         new_pop(i1,:)=child1;
%         new_pop(i2,:)=child2;
%     end
% end
end