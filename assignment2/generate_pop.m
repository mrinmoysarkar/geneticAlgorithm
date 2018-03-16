function pop = generate_pop(n,l,hash,S)
pop = zeros(n,l+1);
pop(:,end) = pop(:,end) + S;
alphabet=[0,1,hash];

for i=1:n
    for j=1:l
        if j==l
            pop(i,j) = alphabet(randi(2,1));
        else
            pop(i,j) = alphabet(randi(3,1));
        end
    end
end
end