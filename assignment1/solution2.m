%% solution2.m
% author: Mrinmoy sarkar
% date: 2/16/2018
% email: msarkar@aggies.ncat.edu


pop_size = 100;
str_len = 20;
pop = round(rand(pop_size,str_len));
pc = 0.7;
pm = 0.001;
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
plot(avg_fit)
hold on
plot(max_fit)
ylim([min(avg_fit)-.01*min(avg_fit),max(max_fit)+.01*max(max_fit)])
maxfit = max(avg_fit)