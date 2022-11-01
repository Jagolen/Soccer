%uppg1a

x = [6 10 8 11 7];
y = [1 -1 1 -1 1];
b1 = 9;
b2 = -1;

xth = x*b2 + b1;
prob = (exp(y.*xth))./(1+exp(y.*xth));
loglike = sum(log(prob));

%uppg1b

rhigh = exp(xth(3))/(1+exp(xth(3)));
rlow = exp(xth(2))/(1+exp(xth(2)));

%uppg1c

xtest = [8 9 11 7 12];
ytest = [1 -1 1 1 -1];
yth = [0 0 0 0 0];


r = 0:0.1:1;
p = zeros(1,length(r));
n = zeros(1,length(r));
fp = zeros(1,length(r));
tp = zeros(1,length(r));

for k = 1:length(r)
    for i = 1:5
        yth(i) = exp(b1 + b2*xtest(i))/(1+exp(b1 + b2*xtest(i)));
        if yth(i) > r(k)
            yth(i) = 1;
            p(k) = p(k) + 1;
            if ytest(i) == 1
                tp(k) = tp(k) + 1;
            end
        else
            yth(i) = -1;
            n(k) = n(k) + 1;
            if ytest(i) == 1
                fp(k) = fp(k) + 1;
            end
        end
    end
    missc = mean(yth ~= ytest)
    if n(k) ~= 0
        fp(k) = fp(k)/n(k)
    end
    if p(k) ~= 0
        tp(k) = tp(k)/p(k)
    end  
end
plot(fp,tp)