clear all;
close all;


A = 2;
D = 2^A;
Out = 1;

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


n = 400;

pop = zeros(n,A+D+Out+1);
S=100;
R=1000;
pop(:,end) = pop(:,end) + S;
hash = 7;

for i=1:n
    for j=1:A+D+Out
        r = rand;
        if j==A+D+Out
            if r<=0.5
                pop(i,j) = 0;
            else
                pop(i,j) = 1;
            end
        else
            if r <= 0.33
                pop(i,j) = 0;
            elseif r > 0.33 && r <= 0.66
                pop(i,j) = 1;
            else
                pop(i,j) = hash;
            end
        end
    end
end

total_ite = 18000;
Cext = 0.05;
Ctax = 0.8;
Cbid = 0.1;
C = 8;
ite = 0;
Pc = 0.6;
Pm = 0.001;

while ite <= total_ite
    for i=1:size(env,1)
        msg = env(i,:);
        pop = match(msg,pop,hash,Cbid,Ctax,Cext,C,R,ite,Pc,Pm);
        ite = ite+1;
    end
end
B = sortrows(round(pop),8,'descend');

B(1:40,:)