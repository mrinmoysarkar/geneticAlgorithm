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
l=A+D+Out;
S=100;
R=1000;
hash = 7;


total_ite = 60000;
Cext = 0.005;
Ctax = 0.85;
Cbid = 0.1;
C = 8;
Pc = 0.6;
Pm = 0.001;

no_of_trial = 20;
pop = generate_pop(n,l,hash,S);
for trial=1:no_of_trial
    ite=0;
    indx1=1;
    total_corect = 0;
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
                avg_score1(indx1) = total_corect;
                percent_hash1(indx1) = count_hash(pop,hash);
                Tsol1(indx1) = noOfCorrectSol(pop,hash);
                indx1 = indx1+1;
                total_corect = 0;
            end
            if mod(ite,5000)==0
                sol_count = count_sol(pop);
            end
        end
    end
    if trial == 1
        avg_score = avg_score1;
        percent_hash = percent_hash1;
        Tsol = Tsol1;
    else
        avg_score = avg_score + avg_score1;
        percent_hash = percent_hash + percent_hash1;
        Tsol = Tsol + Tsol1;
        
    end
    %sol_count
end
sol_count = count_sol(pop);
final_solution = sol_count(1:8,:);
final_solution(:,end+1) = bi2de(final_solution(:,1:2),'left-msb');
final_solution = sortrows(final_solution,10);
for i=1:8
    for j=1:9
        if final_solution(i,j) == hash
            printf('#')
        else
            if j==7 || j==8
                printf('\t')
            end
            printf('%d',final_solution(i,j))
        end
    end
    printf('\n')
end

figure(1)
subplot(311)
plot(avg_score/no_of_trial)
title('average score')
subplot(312)
plot(percent_hash/no_of_trial)
title('percent hash')
subplot(313)
plot(Tsol/no_of_trial)
title('Tsol')