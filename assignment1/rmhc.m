%% rmhc.m
% author: Mrinmoy sarkar
% date: 2/16/2018
% email: msarkar@aggies.ncat.edu

no_of_run = 10;
no_of_iteration = 10000;
pop_size = 1;
str_len = 20;
for kkk=1:2
    if kkk==1
        fit = @fitness1;
    else
        fit = @fitness2;
    end
    
    avg_fit = zeros(no_of_run,100);
    for r=1:no_of_run
        pop = round(rand(pop_size,str_len));
        fit_mat=zeros(1,no_of_iteration);
        fit_mat(1) = fit(pop);
        best_fit = fit_mat(1);
        kk = 1;
        for n=2:no_of_iteration
            i = randi(str_len,1,1);
            pop(i) = xor(pop(i),1);
            fit_mat(n) = fit(pop);
            if fit_mat(n)>=best_fit
                best_fit = fit_mat(n);
            else
                pop(i) = xor(pop(i),1);
            end
            if mod(n,100) == 0
                avg_fit(r,kk) = mean(fit_mat(n-100+1:n));% best_fit;
                kk = kk+1;
            end
        end
    end
    
    if kkk==1
        subplot(211)
        plot(mean(avg_fit))
        xlabel('no of generation(equivalent to GA)')
        ylabel('average fitness')
        title('problem 1')
        ylim([0,22])
    else
        subplot(212)
        plot(mean(avg_fit))
        xlabel('no of generation(equivalent to GA)')
        ylabel('average fitness')
        title('problem 2')
    end
end

