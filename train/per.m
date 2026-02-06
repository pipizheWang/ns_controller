clc;
clear;
close all;

% 读取原始数据
Data1_raw = load_your_csv_matlab("uav0_20260205_000450.csv");
Data2_raw = load_your_csv_matlab("uav1_20260205_000446.csv");

% 统一时间区间
t0 = max(Data1_raw.time(1), Data2_raw.time(1));
t1 = min(Data1_raw.time(end), Data2_raw.time(end));

% 100Hz 插值
Data1_int = interpolation_100hz(t0, t1, Data1_raw);
Data2_int = interpolation_100hz(t0, t1, Data2_raw);

% 参数
g = 9.81;
m_kg = 2.0;

% 推力拟合参数（单位：N）
p_00 = 354.716813;
p_10 = -0.483469;
p_01 = -19.170488;
p_20 = 8.183327e-05;
p_11 = 0.022760;

% ===================== Fa 计算（展开，不封装）=====================
L = numel(Data1_int.time);

pwm = Data1_int.pwm;    % Lx4
vol = Data1_int.vol;    % Lx1
qua = Data1_int.qua;    % Lx4 (x y z w)
acc = Data1_int.acc_imu;% Lx3

% 1) 推力模型：每个电机推力 + 总推力
force1 = p_00 + p_10*pwm(:,1) + p_01*vol + p_20*(pwm(:,1).^2) + p_11*(vol.*pwm(:,1));
force2 = p_00 + p_10*pwm(:,2) + p_01*vol + p_20*(pwm(:,2).^2) + p_11*(vol.*pwm(:,2));
force3 = p_00 + p_10*pwm(:,3) + p_01*vol + p_20*(pwm(:,3).^2) + p_11*(vol.*pwm(:,3));
force4 = p_00 + p_10*pwm(:,4) + p_01*vol + p_20*(pwm(:,4).^2) + p_11*(vol.*pwm(:,4));
thrust = force1 + force2 + force3 + force4; % Lx1, N

% 2) 由四元数得到 body z 轴在 world 的方向 ez_world（对应 R(:,3)）
ez_world = zeros(L,3);
for k = 1:L
    R = rotation_matrix_from_quat_xyzw(qua(k,:)); % 3x3
    ez_world(k,:) = R(:,3).'; % 1x3
end

% 3) 反推气动力 Fa（与Python一致）
% Fa = m*acc_imu - thrust*ez_world - [0 0 -m*g]
Fa = m_kg * acc - thrust .* ez_world - repmat([0, 0, -m_kg*g], L, 1);

% 写回 Data1
Data1 = Data1_int;
Data1.fa_imu = Fa;

% ===================== 工具区分析（建议放断点 / 看变量）=====================
% 常用检查量（你可以在这里逐个画图排查）
% 1) 推力是否量级合理
% 2) ez_world 是否单位向量
% 3) acc_imu 是否已经包含重力（很多IMU加速度是 specific force）
thrust_stats = [min(thrust), mean(thrust), max(thrust)];
ez_norm = vecnorm(ez_world, 2, 2);
fa_norm = vecnorm(Fa, 2, 2);

% 示例：快速看几条曲线
figure; plot(Data1.time, thrust); grid on; xlabel("t"); ylabel("thrust (N)");
figure; plot(Data1.time, ez_norm); grid on; xlabel("t"); ylabel("||R(:,3)||");
figure; plot(Data1.time, Fa); grid on; xlabel("t"); ylabel("Fa (N)"); legend("Fx","Fy","Fz");

% 如果你怀疑“加速度是否含重力”，可以对比：
% hover 时理论上：若acc_imu是specific force（含-重力项已去掉/或反之），公式要改
% 这里给你一个对比版本（不写回Data1，只做对照）
Fa_alt = m_kg * (acc + repmat([0,0,g],L,1)) - thrust .* ez_world;  % 仅用于对照排查
figure; plot(Data1.time, Fa(:,3), Data1.time, Fa_alt(:,3)); grid on;
xlabel("t"); ylabel("Fz (N)"); legend("Fa original Fz","Fa alt Fz");

% ===================== 生成训练对 =====================
[data_input, data_output] = get_data_pair(Data1, Data2_int, "fa_imu");


% ===================== 子函数 =====================

function Data = load_your_csv_matlab(filename)
T = readtable(filename);

tick = T.tick;
time = (tick - tick(1)) / 1e6; % 秒

Data.time = time(:);
Data.pos  = [T.pos_x, T.pos_y, T.pos_z];
Data.vel  = [T.vel_x, T.vel_y, T.vel_z];
Data.acc  = [T.acc_x, T.acc_y, T.acc_z];
Data.qua  = [T.qua_x, T.qua_y, T.qua_z, T.qua_w]; % [x y z w]
Data.pwm  = [T.pwm_1, T.pwm_2, T.pwm_3, T.pwm_4];
Data.vol  = T.voltage;

N = height(T);
Data.tau_u = zeros(N,3);
Data.omega = zeros(N,3);
end

function Data_int = interpolation_100hz(t0, t1, Data)
dt = 0.01;
time = (t0:dt:t1).';
L = numel(time);

pos = zeros(L,3);
vel = zeros(L,3);
acc_imu = zeros(L,3);
tau_u = zeros(L,3);
omega = zeros(L,3);
qua = zeros(L,4);
pwm = zeros(L,4);
vol = zeros(L,1);

x = Data.time;

for i=1:3
    pos(:,i)     = interp1(x, Data.pos(:,i), time, "linear", "extrap");
    vel(:,i)     = interp1(x, Data.vel(:,i), time, "linear", "extrap");
    acc_imu(:,i) = interp1(x, Data.acc(:,i), time, "linear", "extrap");
    tau_u(:,i)   = interp1(x, Data.tau_u(:,i), time, "linear", "extrap");
    omega(:,i)   = interp1(x, Data.omega(:,i), time, "linear", "extrap");
end
for i=1:4
    qua(:,i) = interp1(x, Data.qua(:,i), time, "linear", "extrap");
    pwm(:,i) = interp1(x, Data.pwm(:,i), time, "linear", "extrap");
end
vol(:,1) = interp1(x, Data.vol(:), time, "linear", "extrap");

vel_num   = zeros(L,3);
acc_num   = zeros(L,3);
omega_dot = zeros(L,3);

for i=1:3
    acc_num(3:end-2,i)   = (-vel(5:end,i) + 8*vel(4:end-1,i) - 8*vel(2:end-3,i) + vel(1:end-4,i)) / 12 * 100;
    vel_num(3:end-2,i)   = (-pos(5:end,i) + 8*pos(4:end-1,i) - 8*pos(2:end-3,i) + pos(1:end-4,i)) / 12 * 100;
    omega_dot(3:end-2,i) = (-omega(5:end,i) + 8*omega(4:end-1,i) - 8*omega(2:end-3,i) + omega(1:end-4,i)) / 12 * 100;
end

euler = zeros(L,3);
for k=1:L
    euler(k,:) = qua2euler_deg(qua(k,:));
end

[b,a] = butter(1, 0.1, "low");
acc_filter = zeros(L,3);
for i=1:3
    acc_filter(:,i) = filtfilt(b,a, acc_num(:,i));
end

n = 5;
acc_smooth = movmean(acc_num, n, 1, "Endpoints","shrink");

Data_int.time       = time;
Data_int.pos        = pos;
Data_int.vel        = vel;
Data_int.acc_imu    = acc_imu;
Data_int.vel_num    = vel_num;
Data_int.acc_num    = acc_num;
Data_int.qua        = qua;
Data_int.pwm        = pwm;
Data_int.vol        = vol;
Data_int.euler      = euler;
Data_int.tau_u      = tau_u;
Data_int.omega      = omega;
Data_int.omega_dot  = omega_dot;
Data_int.acc_filter = acc_filter;
Data_int.acc_smooth = acc_smooth;
end

function R = rotation_matrix_from_quat_xyzw(q)
x=q(1); y=q(2); z=q(3); w=q(4);

a = x^2; b = y^2; c = z^2; d = w^2;
e = x*y; f = x*z; g = x*w; h = y*z; i = y*w; j = z*w;

R = zeros(3,3);
R(1,1) = a - b - c + d;
R(1,2) = 2*(e - j);
R(1,3) = 2*(f + i);
R(2,1) = 2*(e + j);
R(2,2) = -a + b - c + d;
R(2,3) = 2*(h - g);
R(3,1) = 2*(f - i);
R(3,2) = 2*(h + g);
R(3,3) = -a - b + c + d;
end

function eul = qua2euler_deg(q_xyzw)
x = q_xyzw(1); y = q_xyzw(2); z = q_xyzw(3); w = q_xyzw(4);

roll  = atan2( 2*(w*x + y*z), 1 - 2*(x^2 + y^2) );
pitch = asin(  2*(w*y - z*x) );
yaw   = atan2( 2*(w*z + x*y), 1 - 2*(y^2 + z^2) );

eul = [rad2deg(roll), rad2deg(pitch), rad2deg(yaw)];
end

function [data_input, data_output] = get_data_pair(D1, D2, typ)
g = 9.81;
L = numel(D1.time);

data_input  = zeros(L, 13, "single");
data_output = zeros(L, 3,  "single");

data_input(:,1:3) = single(D2.pos - D1.pos);
data_input(:,4:6) = single(D2.vel - D1.vel);
data_input(:,13)  = single(2);

Fa = D1.(typ);
data_output(:,:) = single(Fa / g * 1000);
end
