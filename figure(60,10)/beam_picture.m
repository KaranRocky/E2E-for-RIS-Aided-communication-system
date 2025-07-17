
N = 8;
n = 0:1:N-1;
phi = -pi/2:0.01:pi/2;
lambda = 0.01;
d = lambda/2;
theta = 2*d/lambda*sin(phi);
at = 1/sqrt(N)*exp(1i*pi*n.*theta');

%b = randn(1,8)+1i*rand(1,8);
%b = b/sqrt(sum(abs(b).^2));



for i=1:2:16
    gain = t_data(i,:)*at';
    plot(phi,gain)
    hold on
end