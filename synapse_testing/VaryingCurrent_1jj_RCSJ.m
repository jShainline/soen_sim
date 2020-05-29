clc; clearvars; close all;
h_bar=6.62607015e-34/2/pi;
e=1.602e-19;

Area=40;
Ic=Area*1e-6;
C=Area*5e-15;
V_shunt=0.25e-3;
R=V_shunt/Ic;
beta_c=2*e*R*R*C*Ic/h_bar;


dTau=1e-2;
fTau=1e3;
Tau_array=0:dTau:fTau; 
I=0:.2e-9:20e-6;
sai1=ones(1,length(Tau_array)+1)*5;
sai2=ones(1,length(Tau_array)+1)*5;
% I=20e-6*sin(Tau_array);
for j=1:length(Tau_array)
    sai1(j+1) = sai1(j)+sai2(j)*dTau;
    sai2(j+1) = sai2(j)+(-sai2(j)/beta_c-sin(sai1(j))/beta_c+I(j)/Ic/beta_c)*dTau;
end
t_array=Tau_array*(h_bar/(2*e*R*Ic));
dt=dTau*(h_bar/(2*e*R*Ic));
vt=h_bar/2/e*diff(sai1)/dt;

plot(t_array(round(end/2):end)*1e9,I(round(end/2):end)*1e6)
hold on;
plot(t_array(round(end/2):end)*1e9,vt(round(end/2):end)*1e6)
