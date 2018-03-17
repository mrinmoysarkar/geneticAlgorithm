function env = getenvironment(A,D)
dec = (0:(2^(A+D)-1))';
bin = de2bi(dec,'left-msb');
env = zeros(size(bin,1),size(bin,2)+1);
env(1:size(bin,1),1:size(bin,2)) = bin;
for i=1:D
    ind1 = (i-1)*2^D+1;
    ind2 = i*2^D;
    env(ind1:ind2,end)=bin(ind1:ind2,A+i);
end
env = env(randperm(size(env, 1)), :);
end