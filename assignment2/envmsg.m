function msg=env()
msg=round(rand(1,7));
msg(end) = msg(3+bi2de(msg(1:2),'left-msb'));
end