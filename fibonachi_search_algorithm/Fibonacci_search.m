%% Fibonacci_search.m
% author : Mrinmoy Sarkar
% date : 2/3/2018
% email : msarkar@aggies.ncat.edu

clear all;
close all;
warning('off','all')


for l=2:100
    N = 110;
    a = 1;
    b = 10e20;
    del = 0.1;
    m1 = a + (fibonacci(N - 2)/fibonacci(N))*(b-a);
    m2 = a + (fibonacci(N - 1)/fibonacci(N))* (b-a);
    while 1
        f1 = f(m1,l);
        f2 = f(m2,l);
        if f1 < f2
            a = m1;
            m1 = m2;
            N = N-1;
            m2 = a + (fibonacci(N - 1)/fibonacci(N))* (b-a);    
        elseif f1 > f2
            b = m2;
            m2 = m1;
            N = N - 1;
            m1 = a + (fibonacci(N - 2)/fibonacci(N))*(b-a);
        end
        if b-a < del || N < 2
            fprintf('Optimal Population Size for string length l = %d is m* = %d\n', l, round((a+b)/2));
            break;
        end
        %fprintf("%d\n",N);
    end
end