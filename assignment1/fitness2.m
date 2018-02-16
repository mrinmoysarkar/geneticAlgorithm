function f=fitness2(pop)
m = size(pop,1);
l = size(pop,2);
f = zeros(m,1);
for i=1:m
    for j=1:l
        f(i) = f(i)+pop(i,j)*2^(l-j);
    end
end
end