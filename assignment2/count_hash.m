function y=count_hash(pop,hash)
n=size(pop,1);
total_hash = 0;
for i=1:n
    classifier = pop(i,:);
    total_hash = total_hash + noOfHash(classifier,hash);
end
y = (total_hash*100)/(6*n);
end