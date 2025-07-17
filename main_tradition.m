close all; clear; clc;
tic
Iteration=1;    %Number of repeated experiments
dist=[0:10:100];
Sample=1000;

BER = zeros(Sample,length(dist),26);
BER_noRIS = zeros(Sample,length(dist),26);
SER = zeros(Sample,length(dist),26);
SER_noRIS = zeros(Sample,length(dist),26);

B=1;            %Number of base stations
BS_antennas=8;  %Number of base station antennas
User_antennas=2;    %Number of user antennas
P_max=1;        %Maximum power limit of base station
K=1;            %amount of users
P=1;            %Number of subcarriers
R=1;            %RIS quantity
N_ris=256;      %The number of units on each RIS
sigma2=10^(-6); %noise

W = ones(8,1);
Theta = ones(N_ris,N_ris);

fprintf('¾­µä\n');
for s=1:Sample
for a=1:length(dist)
    a=6;
    W = ones(8,1);
    Theta = ones(N_ris,N_ris);
    fprintf('µÚ%dÂÖ\n',a);
    disp(['user location:',num2str(dist(a)),'m'])
    [Dis_BStoRIS, Dis_BStoUser, Dis_RIStoUser]=Position_generate(B,R,K,dist(a));   % Base station RIS user location settings
    for b=1:Iteration       
        large_fading_AI=3;                             %channel attenuation 3 
        large_fading_DI=2;
        %[ H_bkp,F_rkp,G_brp ] = Channel_generate(B,R,K,P,N_ris,BS_antennas,User_antennas,large_fading_AI,large_fading_DI,Dis_BStoRIS, Dis_BStoUser,Dis_RIStoUser);     
        %[ H_bkp,F_rkp,G_brp ] = Channel_read_Quadriga(dist(a),B,R,K,P,N_ris,BS_antennas,User_antennas,large_fading_AI,large_fading_DI);     
        [ H_bkp,F_rkp,G_brp ] = Channel_read(s,B,R,K,P,N_ris,BS_antennas,User_antennas,large_fading_AI,large_fading_DI,Dis_BStoRIS, Dis_BStoUser,Dis_RIStoUser);
        t1=clock;
        %[W,Theta,R_sum_1bit(a,b)]=MyAlgorithm_1bit(B,BS_antennas,User_antennas,P_max,K,P,R,N_ris,sigma2,H_bkp,F_rkp,G_brp);
        %[W,Theta,R_sum_2bit(a,b)]=MyAlgorithm_2bit(B,BS_antennas,User_antennas,P_max,K,P,R,N_ris,sigma2,H_bkp,F_rkp,G_brp,W,Theta);             
        
        [ W,Theta,R_sum_InFbit(a,b)]=MyAlgorithm_InFbit(B,BS_antennas,User_antennas,P_max,K,P,R,N_ris,sigma2,H_bkp,F_rkp,G_brp,W,Theta);
        t2=clock;
        etime(t2,t1)
        [Ser, Ber] = Bler_calculate(B,BS_antennas,User_antennas,P_max,K,P,R,N_ris,sigma2,H_bkp,F_rkp,G_brp,W,Theta);
        BER(s,a,:) = Ber;
        SER(s,a,:) = Ser;
        [Ser_noRIS, Ber_noRIS] = Bler_calculate_noRIS(B,BS_antennas,User_antennas,P_max,K,P,R,N_ris,sigma2,H_bkp);
        BER_noRIS(s,a,:) = Ber_noRIS;
        SER_noRIS(s,a,:) = Ser_noRIS;
%        R_sum(a,b)=MyAlgorithm(B,BS_antennas,User_antennas,P_max,K,P,R,N_ris,sigma2,H_bkp,F_rkp,G_brp,W,Theta);
%        R_sum_sub(a,b)=MyAlgorithm_Sub(B,BS_antennas,P_max,K,P,R,N_ris,sigma2,H_bkp,F_rkp,G_brp);
%        R_sum_bas(a,b)=MyAlgorithm_Bas(B,BS_antennas,User_antennas,P_max,K,P,R,N_ris,sigma2,H_bkp,F_rkp,G_brp)
%        R_sum_ZF(a,b)=MyAlgorithm_ZF(B,BS_antennas,P_max,K,P,sigma2,H_bkp);
%        R_sum_noRIS(a,b)=MyAlgorithm_noRIS(B,BS_antennas,User_antennas,P_max,K,P,R,N_ris,sigma2,H_bkp,F_rkp,G_brp); 
    end
end
end
figure(1)
semilogy(dist,BER(:,tsnr),'--o','LineWidth',1.5);
hold on
semilogy(dist,BER_noRIS(:,tsnr),'-*','LineWidth',1.5);
semilogy(dist,a(:,tsnr),'-V','LineWidth',1.5);
semilogy(dist,b(:,tsnr),'-^','LineWidth',1.5);
xlabel('Distance L/m')
ylabel('BER')
legend('With RIS','Without RIS')
%save(['figure(60,10)/Tradition_Ber_',num2str(N_ris),'.mat'], 'BER')
%save figure(60,10)/Tradition_Ber_0.mat BER_noRIS