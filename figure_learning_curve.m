clc
load('figure(60,10)/R_Error_3.mat')
a1 = R_Error(1:11:330,:);
a2 = R_Error(2:11:330,:);
a3 = R_Error(10:11:330,:);
a4 = R_Error(11:11:330,:);
a = [a1;a2;a3;a4];
a = R_Error(1:1:360,1:150);

b = mean(a,1);
c = std(a,1);
d = b+c;

x = 1:150;        
xconf = [x x(end:-1:1)] ;%a round trip         
yconf = [b-c d(end:-1:1)];%015 is the strip width.,Changing to a matrix will have different widths
p = fill(xconf,yconf,'r','FaceColor',[0 191 255]/255,'EdgeColor','none');%FaceColor is the fill color, EdgeColor is the border color

hold on;
plot(b,'b--','linewidth',1.2)

load('figure(60,10)/R_Error_1.mat')
a = R_Error(1:1:360,1:150);
b = mean(a,1);
c = std(a,1);
d = b+c;
x = 1:150;            
xconf = [x x(end:-1:1)] ;% a round trip        
yconf = [b-c d(end:-1:1)];%0.15 It is the strip width. If you change it to a matrix, it will have different widths.
p = fill(xconf,yconf,'b','FaceColor',[0 238 118]/255,'EdgeColor','none');%FaceColor is the fill color, EdgeColor is the border color
plot(b,'g--','linewidth',1.2,'color',[0 100 0]/255)

load('figure(60,10)/R_Error_5.mat')
a = R_Error(1:1:360,1:150);
b = mean(a,1);
c = std(a,1);
d = b+c;
x = 1:150;          
xconf = [x x(end:-1:1)] ;%a round trip         
yconf = [b-c d(end:-1:1)];%0.15 is the strip width. If you change it to a matrix, it will have different widths.
p = fill(xconf,yconf,'g','FaceColor',[205 85 85]/255,'EdgeColor','none');%FaceColor is the fill color, EdgeColor is the border color
plot(b,'k--','linewidth',1.2)

legend('One std - fixed noise power','Mean - fixed noise power','One std - adjustable noise power','Mean - adjustable noise power','One std - transfer learning','Mean  - transfer learning','FontSize',11)
xlabel('Epoch');
ylabel('Loss value');
grid on;

