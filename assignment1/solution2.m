%% solution2.m
% author: Mrinmoy sarkar
% date: 2/16/2018
% email: msarkar@aggies.ncat.edu

Pcc=[0 .2 .5 .7 .9 1];
Pmm=[0 .001 .005 .009 .01 .09];
popSize=[100 500 1000 1500 2000 3000];
for kk=1:3
    for ii=1:length(Pcc)
        if kk==1
            pop_size = popSize(ii);
            pc = 0.7;
            pm = 0.001;
        elseif kk==2
            pop_size = 100;
            pc = Pcc(ii);
            pm = 0.001;
        else
            pop_size = 100;
            pc = 0.7;
            pm = Pmm(ii);
        end
        str_len = 20;
        pop = round(rand(pop_size,str_len));
        
        gen_no = 100;
        
        fit = @fitness2;
        
        fit_of_gen = zeros(pop_size,gen_no);
        
        for i=1:gen_no
            fit_of_gen(:,i)=fit(pop);
            pop = reproduction(pop,fit);
            pop =  cross(pop,pc);
            pop = mutation(pop,pm);
        end
        
        avg_fit = mean(fit_of_gen);
        max_fit = max(fit_of_gen);
        if kk==1
            subplot(321)
            plot(avg_fit)
            xlabel('generation')
            ylabel('average fitness')
            title('max fitness vs gen for varying Pop size')
            hold on
            subplot(322)
            plot(max_fit)
            xlabel('generation')
            ylabel('maximum fitness')
            title('max fitness vs gen for varying pop size')
            hold on
        elseif kk==2
            subplot(323)
            plot(avg_fit)
            xlabel('generation')
            ylabel('average fitness')
            title('avg fitness vs gen for varying Pc')
            hold on
            subplot(324)
            plot(max_fit)
            xlabel('generation')
            ylabel('maximum fitness')
            title('max fitness vs gen for varying Pc')
            hold on
        else
            subplot(325)
            plot(avg_fit)
            xlabel('generation')
            ylabel('average fitness')
            title('avg fitness vs gen for varying Pm')
            hold on
            subplot(326)
            plot(max_fit)
            xlabel('generation')
            ylabel('maximum fitness')
            title('max fitness vs gen for varying Pm')
            hold on
        end
        ylim([min(avg_fit)-.01*min(avg_fit),max(max_fit)+.1*max(max_fit)])
        %maxfit = max(avg_fit)
    end
end