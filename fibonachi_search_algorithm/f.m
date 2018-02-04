function f=f(m,l)
s = 0;
for j=1:l
    s = s + nchoosek(l,j)*2^j*(1-(1-.5^j)^m);
end
f=(s-2^l)/m;
end