function n=noOfHash(classifier,hash)
n=0;
for i=1:length(classifier)-2
    if hash == classifier(i)
        n = n+1;
    end
end
end