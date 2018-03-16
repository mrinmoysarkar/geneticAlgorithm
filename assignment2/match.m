function y = match(msg,pop,hash)
n = size(pop,1);
y = ones(n,1);
for j=1:n
    classifier = pop(j,:);
    for i=1:length(msg)-1
        if msg(i) ~= classifier(i) && classifier(i) ~= hash
            y(j) = 0;%no match
            break;
        end
    end
end
end
