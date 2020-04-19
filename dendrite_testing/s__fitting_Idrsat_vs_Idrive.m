%% initialize
clc
% clear all
% close all

[fontName,fontSize,fontSize_legend,bRGY,scrsz] = f_plotting;
p = f_physicalConstants;

%% input data
I_drive_vec = 19:30;
I_di_sat_vec = [10.6982 11.712 12.8058 13.9797 15.2335 16.6742 17.5057 17.5323 18.0127 18.0393 18.0393 18.0660];

%% fit initial portion to line
initial_linear_index = 1;
final_linear_index = 8;
[I_di_sat_fit] = polyfit(I_drive_vec(initial_linear_index:final_linear_index),I_di_sat_vec(initial_linear_index:final_linear_index),1);

I_drive_vec_dense = linspace(I_drive_vec(initial_linear_index),I_drive_vec(final_linear_index),100);
I_di_sat_vec_dense = polyval(I_di_sat_fit,I_drive_vec_dense);


%% plot

figure('OuterPosition',[0 0 scrsz(3) scrsz(4)]);
plot(I_drive_vec,I_di_sat_vec,'Color',bRGY(3,:),'LineStyle','-','LineWidth',3)
hold on
plot(I_drive_vec_dense,I_di_sat_vec_dense,'Color',bRGY(8,:),'LineStyle','-','LineWidth',3)
lgd = legend('wr','fit');
lgd.FontSize = fontSize_legend;
ylabel('I_{di}^{sat} [\mu A]','FontSize',fontSize,'FontName','Times')
xlabel('I_{drive} [\mu A]','FontSize',fontSize,'FontName','Times')
set(gca,'FontSize',fontSize,'FontName',fontName)
title(sprintf('I_{di}^{sat} = %g * I_{drive} + %g',I_di_sat_fit(1),I_di_sat_fit(2)),'FontSize',fontSize,'FontName','Times')
% xlim([times(1) times(end)]*1e9)
% xlim([-1 15])
% ylim([0 1.1*max(I_si_wr)*1e6])