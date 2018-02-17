%% square wave from sine wave visualization
% author: Mrinmoy Sarkar
% Date: 2/15/2018
% email: msarkar@aggies.ncat.edu
%%
clear all;
close all;

N=1024;
t=linspace(0,0.04,N);
x = square(2*pi*50*t);
subplot(211)
plot(t,x,'*')
ylim([min(x)-1 max(x)+1])

coeff = fft(x);
N = length(x);
NN = 16;
for L=linspace(1,N,NN)
    x_new = zeros(1,N);
    for n=1:N
        for k=1:L
            x_new(n) = x_new(n) + (1/L)*coeff(k)*exp(1i*2*pi*(k-1)*(n-1)/N);
        end
    end
    hold on;
    subplot(211)
    plot(t,real(x_new))
    ylim([min(real(x_new))-2 max(real(x_new))+2])
    pause(1)
end
subplot(212)
plot(t,x,'b*')
hold on
subplot(212)
plot(t,real(x_new),'r')
ylim([min(real(x_new))-1 max(real(x_new))+1])
legend('original','reconstructed')

