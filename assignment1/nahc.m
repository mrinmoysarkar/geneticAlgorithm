%% nahc.m
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
        kk = 1;
        fit_mat=zeros(1,no_of_iteration);
        n=1;
        while 1
            pop = round(rand(pop_size,str_len));
            fit_mat(n) = fit(pop);
            if n == no_of_iteration
                break
            end
            best_fit = fit_mat(n);
            if mod(n,100) == 0
                avg_fit(r,kk) = best_fit;
                kk = kk+1;
            end
            n = n+1;
            for i = 1:str_len
                pop(i) = xor(pop(i),1);
                fit_mat(n) = fit(pop);
                if fit_mat(n)>=best_fit
                    best_fit = fit_mat(n);
                else
                    pop(i) = xor(pop(i),1);
                end
                if mod(n,100) == 0
                    avg_fit(r,kk) = best_fit;
                    kk = kk+1;
                end
                if n == no_of_iteration
                    break
                end
                n=n+1;
            end
        end
    end
    
    if kkk==1
        subplot(211)
        plot(mean(avg_fit))
        xlabel('no of generation(equivalent to GA)')
        ylabel('average fitness')
        title('problem 1')
    else
        subplot(212)
        plot(mean(avg_fit))
        xlabel('no of generation(equivalent to GA)')
        ylabel('average fitness')
        title('problem 2')
    end
end