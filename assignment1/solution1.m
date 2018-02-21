%% solution1.m
% author: Mrinmoy sarkar
% date: 2/16/2018
% email: msarkar@aggies.ncat.edu


Pcc=[0 .2 .5 .7 .9 1];
Pmm=[0 .001 .005 .009 .01 .09];
for kk=1:2
    for ii=1:length(Pcc)
        
        pop_size = 100;
        str_len = 20;
        pop = round(rand(pop_size,str_len));
        if kk==1
            pc = Pcc(ii);
            pm = 0.001;
        else
            pc=0.7;
            pm=Pmm(ii);
        end
        
        gen_no = 20;
        
        fit = @fitness1;
        
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
            subplot(121)
        else
            subplot(122)
        end
        plot(avg_fit)
        hold on
        
        %plot(max_fit)
        %hold on
        ylim([min(avg_fit)-3,max(max_fit)+3])
        %maxfit = max(avg_fit)
    end
    if kk==1
        xlabel('generation')
        ylabel('average fitness')
        legend('Pc=0,Pm=.001','Pc=.2,Pm=.001','Pc=.5,Pm=.001','Pc=.7,Pm=.001','Pc=.9,Pm=.001','Pc=1,Pm=.001')
    else
        xlabel('generation')
        ylabel('average fitness')
        legend('Pc=.7,Pm=0','Pc=.7,Pm=.001','Pc=.7,Pm=.005','Pc=.7,Pm=.009','Pc=.7,Pm=.01','Pc=.7,Pm=.09')
    end
end