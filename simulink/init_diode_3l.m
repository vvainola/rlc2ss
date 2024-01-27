clc;
L_conv = 100e-5;
L_dc = 100e-5;
L_dc_src = 100e-4;
L_grid = 100e-5;
L_src = 100e-5;
C_dc = 10e-3;
C_f = 1e-5;
R_conv = 1e-3;
R_dc_p = 100;
R_dc = 1e-3;
R_dc_src_p = 1;
R_dc_src = 10;
R_f = 1e-3;
R_grid = 1e-3;
R_src = 1e-3;

freq = 50;
V_dc_src = 500;
initial_angle = 180;
amplitude = 400;
asd = sim('test_diode_lcl');

%iconv, dc
data = [asd.logsout{1}.Values.Data, asd.logsout{2}.Values.Data];
writematrix(data, 'simulink.csv')
%writematrix(asd.logsout{3}.Values.Data, 'simulink_n_cap.csv')