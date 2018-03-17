clear all;
close all;

A = 2;
D = 2^A;
Out = 1;
n = 400;
l=A+D+Out;
S=100;
R=1000;
hash = 7;
total_ite = 500000;
Cext = 0.05;
Ctax = 0.8;
Cbid = 0.1;
C = 8;
Pc = 0.6;
Pm = 0.001;
no_of_trial = 10;

env = getenvironment(A,D);
init_pop = generate_pop(n,l,hash,S);
pop = init_pop;
for trial=1:no_of_trial
    pop = init_pop;%generate_pop(n,l,hash,S);
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
    fprintf("trial no.: %d \n", trial);
end
sol_count = count_sol(pop);
final_solution = sol_count(1:8,:);
final_solution(:,end+1) = bi2de(final_solution(:,1:2),'left-msb');
final_solution = sortrows(final_solution,10);
for i=1:8
    for j=1:9
        if final_solution(i,j) == hash
            fprintf('#')
        else
            if j==7 || j==8 || j==9
                fprintf('\t')
            end
            fprintf('%d',final_solution(i,j))
        end
    end
    fprintf('\n')
end

figure(1)
plot(avg_score/no_of_trial)
title('average score')
xlabel('iteration(x100)')
ylabel('average score')
figure(2)
plot(percent_hash/no_of_trial)
title('% hash')
xlabel('iteration(x100)')
ylabel('% hash')
figure(3)
plot(Tsol/no_of_trial)
title('Tsol')
xlabel('iteration(x100)')
ylabel('Tsol')