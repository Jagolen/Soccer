x = [1 2; 3 4; 3 3; 1 1];
xt = transpose(x);
xtx = xt*x;
xinv = xtx^-1;
y = [2; 8; 5; 3];
xty = xt*y;
th = xinv*xty