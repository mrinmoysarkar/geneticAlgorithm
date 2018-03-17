function y=count_sol(pop)
k=1;
while ~isempty(pop)
    classifier = pop(1,:);
    s = sum(classifier(1:7)==pop(:,1:7),2);
    indx = find(s==7);
    total_s = sum(pop(indx,end));
    total_copy = length(indx);
    pop(indx,:)=[];
    classifier(end) = total_s;
    y(k,:)=[classifier total_copy];
    k = k+1;
end
y=sortrows(round(y),8,'descend');
end