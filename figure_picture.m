
BER_noRIS = load('figure(60,10)/Tradition_Ber_0.mat');
BER_noRIS = squeeze(mean(BER_noRIS.BER_noRIS,1));

BER128_trad = load('figure(60,10)/Tradition_Ber_128.mat');
BER128_trad = squeeze(mean(BER128_trad.BER,1));

BER256_trad = load('figure(60,10)/Tradition_Ber_256.mat');
BER256_trad = squeeze(mean(BER256_trad.BER,1));

BER128_e2e = load('figure(60,10)/E2E_Ber_128.mat');
BER128_e2e = squeeze(mean(BER128_e2e.Ber,1));

BER256_e2e = load('figure(60,10)/E2E_Ber_256.mat');
BER256_e2e = squeeze(mean(BER256_e2e.Ber,1));

figure(1)
x = 0:10:100;
k = 5;
semilogy(x,BER_noRIS(:,k),'r-o','LineWidth',1.7,'color',[0.13 0.55 0.13]);
hold on;
semilogy(x,BER128_trad(:,k),'b-s','LineWidth',1.7); 
%semilogy(x,BER128_trad(:,k),'m--p','LineWidth',1.7); 
semilogy(x,BER128_e2e(:,k),'r-v','LineWidth',1.7);%,'color',[255 153 18]/255);
semilogy(x,BER256_trad(:,k),'b--o','LineWidth',1.7);
semilogy(x,BER256_e2e(:,k),'r--^','LineWidth',1.7); 
grid on
axis([0 100 1e-6 1])
%title(['RIS-aided communication system  SNR=-1dB'])
xlabel('{\it L} (m)')
ylabel('BER')
legend('Without RIS','128 - Alternating Scheme [6]','128 - Proposed E2E Scheme','256 - Alternating Scheme [6]','256 - Proposed E2E Scheme')
%legend('Without RIS','Ideal RIS case in [6]','Continuous phase shift in [6]','Proposed E2E learning scheme',...
%'256 Elements RIS - Alternating Scheme[6]',...
%'256 Elements RIS - Proposed E2E Scheme')

figure(2)
x = -5:20;
k = 3;
semilogy(x,BER_noRIS(k,:),'r-o','LineWidth',1.7,'color',[0.13 0.55 0.13]);
hold on;
semilogy(x,BER128_trad(k,:),'b-s','LineWidth',1.7); 
%semilogy(x,BER128_trad(k,:),'m--p','LineWidth',1.7); 
semilogy(x,BER128_e2e(k,:),'r-v','LineWidth',1.7);
semilogy(x,BER256_trad(k,:),'b--o','LineWidth',1.7);%,'color',[255 153 18]/255);
semilogy(x,BER256_e2e(k,:),'r--^','LineWidth',1.7); 
grid on
axis([-5 20 0.000001 1])
%title('RIS-aided communication system L=20m')
xlabel('SNR (dB)')
ylabel('BER')
legend('Without RIS','128 - Alternating Scheme [6]','128 - Proposed E2E Scheme','256 - Alternating Scheme [6]','256 - Proposed E2E Scheme')

%legend('Without RIS','Ideal RIS case in [6]','Continuous phase shift in [6]','Proposed E2E learning scheme',...
%'256 Elements RIS - Alternating Scheme[6]',...
%'256 Elements RIS - Proposed E2E Scheme')