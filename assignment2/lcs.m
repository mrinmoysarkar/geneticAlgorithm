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

alphabet=[0,1,hash];

for i=1:n
    for j=1:A+D+Out
        if j==A+D+Out
            pop(i,j) = alphabet(randi(2,1));
        else
            pop(i,j) = alphabet(randi(3,1));
        end
    end
end

total_ite = 60000;
Cext = 0.005;
Ctax = 0.85;
Cbid = 0.1;
C = 8;
ite = 0;
Pc = 0.6;
Pm = 0.001;
total_corect = 0;
indx1=1;
while ite <= total_ite
    if mod(ite,10000)==0
        env = env(randperm(size(env, 1)), :);
    end
    for i=1:size(env,1)
        msg = env(i,:);
        m = match(msg,pop,hash);
        [pop,corect] = clearinghouse(pop,m,Cext,Ctax,Cbid,R,msg,hash);
        ite = ite+1;
        total_corect = total_corect + corect;
        if mod(ite,25)==0
            pop = ga(pop,Pc,Pm,hash);
        end
        if mod(ite,100)==0
            avg_score(indx1) = total_corect;
            percent_hash(indx1) = count_hash(pop,hash);
            Tsol(indx1) = noOfCorrectSol(pop,hash);
            indx1 = indx1+1;
            total_corect = 0;
        end
        if mod(ite,5000)==0
            sol_count = count_sol(pop)
        end
    end
end

figure(1)
subplot(311)
plot(avg_score)
title('average score')
subplot(312)
plot(percent_hash)
title('percent hash')
subplot(313)
plot(Tsol)
title('Tsol')