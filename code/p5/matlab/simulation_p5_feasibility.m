%% simulate_problem5_feasibility.m
% 题目A - 问题5：基于 check_uav_feasibility.py 的“存在性”结果，生成多机多弹可视化动画
% 功能：
% 1) 同时展示 3 枚导弹(M1~M3) 直指诱饵(原点)飞行；5 架无人机(FY1~FY5) 选取一个可行干扰方案。
% 2) 对每架无人机，自动搜索“最早可行”的干扰方案（对应脚本中的充分条件）：
%    - 在某时刻 T，选取 LOS(原点<-导弹) 上某点，使得：
%        a) 水平可达：|U0->Pxy| = v*T, v∈[70,140]m/s
%        b) 垂直/时序：存在引信 f≤T，使 z_det = z_uav - 0.5*g*f^2，且 0≤z_det≤z_missile(T)
%    - 满足则确定：无人机速度 v、航向、释放时刻 tau=T-f、释放点与起爆点。
% 3) 动画绘制：导弹/无人机轨迹与当前位置、各 UAV 的 LOS、云团球体（半径10m，起爆后20s，下沉3m/s）。
% 4) 自动保存 GIF/MP4 到本目录 output/。

clear; clc; close all;

%% ===================== 常量与参数 =====================
g        = 9.8;       % m/s^2
vM       = 300.0;     % 导弹速度
v_min    = 70.0;      % 无人机最小速
v_max    = 140.0;     % 无人机最大速
v_sink   = 3.0;       % 云团下沉速度
R_cloud  = 10.0;      % 云团有效半径
T_eff    = 20.0;      % 起爆后有效时间

% 诱饵与真目标说明：本可视化采用诱饵(原点)为 LOS 终点，以与 feasibility 脚本保持一致。
decoy = [0,0,0];      % 诱饵坐标 (0,0,0)

%% ===================== 初始态（与 python 一致） =====================
% 导弹初始位置
M0s = [
    20000,     0, 2000;   % M1
    19000,   600, 2100;   % M2
    18000,  -600, 1900;   % M3
];

% 无人机初始位置（等高飞行）
U0s = [
    17800,      0, 1800;  % FY1
    12000,   1400, 1400;  % FY2
    6000,   -3000,  700;  % FY3
    11000,   2000, 1800;  % FY4
    13000,  -2000, 1300;  % FY5
];

numM = size(M0s,1);
numU = size(U0s,1);

% 预计算导弹单位方向与撞击时间
uM = zeros(numM,3);
tImpact = zeros(numM,1);
for i = 1:numM
    m0 = M0s(i,:);
    uM(i,:) = -m0 / norm(m0);
    tImpact(i) = norm(m0) / vM;
end

%% ===================== 为每架无人机搜索一个“最早可行方案” =====================
sol.has = false(numU,1);
sol.mIdx = nan(numU,1);
sol.T    = nan(numU,1);
sol.f    = nan(numU,1);
sol.v    = nan(numU,1);
sol.head = nan(numU,3);
sol.Prel = nan(numU,3);
sol.Ept  = nan(numU,3);

timeSamples   = 200;   % 与 python 一致的采样量级
heightSamples = 120;

for ui = 1:numU
    U0 = U0s(ui,:);
    zU = U0(3);

    for mi = 1:numM
        m0 = M0s(mi,:);
        tImp = tImpact(mi);
        % 在 (0, tImp] 采样
        found = false;
        for iT = 1:timeSamples
            T = tImp * iT / timeSamples;
            Mt = m0 + vM * T * uM(mi,:);
            mx = Mt(1); my = Mt(2); mz = Mt(3);
            if mz <= 0, continue; end

            z_upper = min(mz, zU);
            if z_upper <= 0, continue; end

            lam_upper = z_upper / mz;
            for ik = 1:heightSamples
                lam = lam_upper * ik / heightSamples;
                z_det = lam * mz;
                % 垂直/时序：f <= T
                f = sqrt(max(0, 2.0 * (zU - z_det) / g));
                if f > T, continue; end

                % 水平可达：|U0->(lam*mx, lam*my)| = v*T, v∈[v_min,v_max]
                x_los = lam * mx; y_los = lam * my;
                r = hypot(x_los - U0(1), y_los - U0(2));
                v_req = r / max(T, eps);
                if v_req < v_min - 1e-6 || v_req > v_max + 1e-6
                    continue;
                end

                % 记录方案（最早 T）
                head_xy = [x_los - U0(1), y_los - U0(2), 0];
                if norm(head_xy(1:2)) < 1e-9, continue; end
                head = head_xy / norm(head_xy);
                tau  = T - f;                       % 释放时刻
                P_rel = U0 + v_req * tau * head;    % 释放点（等高）
                Ept   = [x_los, y_los, z_det];      % 起爆点

                sol.has(ui) = true;
                sol.mIdx(ui)= mi;
                sol.T(ui)   = T;
                sol.f(ui)   = f;
                sol.v(ui)   = v_req;
                sol.head(ui,:)= head;
                sol.Prel(ui,:)= P_rel;
                sol.Ept(ui,:)= Ept;
                found = true;
                break;
            end
            if found, break; end
        end
        if found, break; end
    end
end

%% ===================== 仿真时间轴 =====================
t0 = 0.0;
t1 = max([tImpact; sol.T(~isnan(sol.T)) + T_eff]);
dt = 0.02;                % 动画时间步
t  = (t0:dt:t1)';
N  = numel(t);

%% ===================== 轨迹预计算 =====================
% 导弹轨迹（超出撞击时刻后保持在原点）
M = nan(N,3,numM);
for mi = 1:numM
    for k = 1:N
        tk = t(k);
        if tk <= tImpact(mi)
            M(k,:,mi) = M0s(mi,:) + vM * tk * uM(mi,:);
        else
            M(k,:,mi) = decoy;  % 撞击后视为到达原点
        end
    end
end

% 无人机轨迹（无限延续直线、等高）
U = nan(N,3,numU);
for ui = 1:numU
    U0 = U0s(ui,:);
    if sol.has(ui)
        vU = sol.v(ui); hd = sol.head(ui,:);
    else
        % 若未找到可行方案，令其静止以便可视化（或给定最小速朝向原点）
        vU = 0; hd = [0 0 0];
    end
    for k = 1:N
        tk = t(k);
        U(k,:,ui) = U0 + vU * tk * hd;
        U(k,3,ui) = U0(3);       % 等高
    end
end

% 云团中心（每个 UAV 一团，若有方案则在 [T, T+T_eff] 有效）
C = nan(N,3,numU);
for ui = 1:numU
    if ~sol.has(ui), continue; end
    Texp = sol.T(ui);
    Ept  = sol.Ept(ui,:);
    idxCloud = (t >= Texp) & (t <= Texp + T_eff);
    C(idxCloud,1,ui) = Ept(1);
    C(idxCloud,2,ui) = Ept(2);
    C(idxCloud,3,ui) = Ept(3) - v_sink * (t(idxCloud) - Texp);
end

%% ===================== 坐标轴范围 =====================
allXYZ = [];
for mi = 1:numM, allXYZ = [allXYZ; squeeze(M(:,:,mi))]; end %#ok<AGROW>
for ui = 1:numU, allXYZ = [allXYZ; squeeze(U(:,:,ui))]; end %#ok<AGROW>
for ui = 1:numU, if sol.has(ui), allXYZ = [allXYZ; sol.Prel(ui,:); sol.Ept(ui,:)]; end, end %#ok<AGROW>

xl = [min(allXYZ(:,1))-1500, max(allXYZ(:,1))+1500];
yl = [min(allXYZ(:,2))-1500, max(allXYZ(:,2))+1500];
zl = [0, max(allXYZ(:,3))+500];

%% ===================== 图形对象初始化 =====================
[su, sv] = sphere(24);
colM = lines(numM);        % 导弹颜色
colU = lines(numU+2);      % 无人机颜色

fig = figure('Color','w','Position',[60 60 1080 680]);
ax  = axes(fig); hold(ax,'on'); grid(ax,'on'); box(ax,'on'); axis(ax,'equal');
xlabel(ax,'X (m)'); ylabel(ax,'Y (m)'); zlabel(ax,'Z (m)');
title(ax, '问题5：多机多弹可行干扰方案（三维动画）');
xlim(ax,xl); ylim(ax,yl); zlim(ax,zl); view(ax, 34, 20);

% 诱饵
plot3(ax, decoy(1),decoy(2),decoy(3), 'kx','MarkerSize',8,'LineWidth',1.2);

% 轨迹与当前位置句柄
hMpath = gobjects(numM,1); hMnow = gobjects(numM,1);
for mi = 1:numM
    hMpath(mi) = plot3(ax, nan, nan, nan, '-', 'LineWidth',1.2, 'Color',colM(mi,:));
    hMnow(mi)  = plot3(ax, nan, nan, nan, '^', 'MarkerSize',8, 'MarkerFaceColor',colM(mi,:), 'Color',colM(mi,:));
end

hUpath = gobjects(numU,1); hUnow = gobjects(numU,1);
for ui = 1:numU
    hUpath(ui) = plot3(ax, nan, nan, nan, '-', 'LineWidth',1.2, 'Color',colU(ui,:));
    hUnow(ui)  = plot3(ax, nan, nan, nan, 'o', 'MarkerSize',7, 'MarkerFaceColor',colU(ui,:), 'Color',colU(ui,:));
end

% 各 UAV 的 LOS 与云团
hLOS  = gobjects(numU,1);
hCloud= gobjects(numU,1);
for ui = 1:numU
    hLOS(ui)   = plot3(ax, nan, nan, nan, '-.', 'LineWidth',1.0, 'Color',[0.85 0.2 0.2]);
    hCloud(ui) = surf(ax, nan(2), nan(2), nan(2), 'EdgeAlpha',0.15, 'FaceAlpha',0.25, 'FaceColor',colU(ui,:));
end

% 释放/起爆点
for ui = 1:numU
    if sol.has(ui)
        plot3(ax, sol.Prel(ui,1), sol.Prel(ui,2), sol.Prel(ui,3), 'ks','MarkerSize',7,'LineWidth',1.0);
        plot3(ax, sol.Ept(ui,1),  sol.Ept(ui,2),  sol.Ept(ui,3),  'kd','MarkerSize',7,'LineWidth',1.0);
    end
end

legend(ax, {'诱饵(原点)'}, 'Location','northeastoutside');

% 右上角信息
txtInfo = text(ax, xl(2), yl(2), zl(2), '', 'HorizontalAlignment','right','VerticalAlignment','top', ...
    'FontSize',10,'FontWeight','bold');

%% ===================== 动画循环 =====================
outDir = fullfile(fileparts(mfilename('fullpath')), 'output');
if ~exist(outDir,'dir'), mkdir(outDir); end
gifName = fullfile(outDir, 'problem5_feasibility.gif');
mp4Name = fullfile(outDir, 'problem5_feasibility.mp4');
saveGIF = true; saveMP4 = true; fps = 25;
if saveMP4
    vw = VideoWriter(mp4Name, 'MPEG-4'); vw.FrameRate = fps; open(vw);
end

for k = 1:N
    % 导弹
    for mi = 1:numM
        set(hMpath(mi), 'XData', M(1:k,1,mi), 'YData', M(1:k,2,mi), 'ZData', M(1:k,3,mi));
        set(hMnow(mi),  'XData', M(k,1,mi),   'YData', M(k,2,mi),   'ZData', M(k,3,mi));
    end

    % 无人机
    for ui = 1:numU
        set(hUpath(ui), 'XData', U(1:k,1,ui), 'YData', U(1:k,2,ui), 'ZData', U(1:k,3,ui));
        set(hUnow(ui),  'XData', U(k,1,ui),   'YData', U(k,2,ui),   'ZData', U(k,3,ui));
    end

    % LOS 与云团
    for ui = 1:numU
        if ~sol.has(ui)
            set(hLOS(ui),   'Visible','off');
            set(hCloud(ui), 'Visible','off');
            continue;
        end
        mi = sol.mIdx(ui);
        % 当前 LOS：导弹当前位置 -> 诱饵
        set(hLOS(ui), 'XData',[M(k,1,mi) decoy(1)], 'YData',[M(k,2,mi) decoy(2)], 'ZData',[M(k,3,mi) decoy(3)], ...
            'Visible','on');

        % 云团（若在有效时间窗）
        Texp = sol.T(ui);
        if t(k) >= Texp && t(k) <= Texp + T_eff
            Ck = squeeze(C(k,:,ui));
            set(hCloud(ui), 'XData', R_cloud*su + Ck(1), ...
                            'YData', R_cloud*sv + Ck(2), ...
                            'ZData', R_cloud*su*0 + Ck(3) + R_cloud*sv*0, 'Visible','on');
        else
            set(hCloud(ui), 'Visible','off');
        end
    end

    % 信息文本
    txt = sprintf('t = %.2f s\n可行方案数量：%d / %d', t(k), sum(sol.has), numU);
    set(txtInfo, 'String', txt);

    drawnow;

    fr = getframe(fig);
    if saveGIF
        [A,map] = rgb2ind(fr.cdata, 256);
        if k == 1
            imwrite(A, map, gifName, 'gif', 'LoopCount', Inf, 'DelayTime', 1/fps);
        else
            imwrite(A, map, gifName, 'gif', 'WriteMode', 'append', 'DelayTime', 1/fps);
        end
    end
    if saveMP4
        writeVideo(vw, fr);
    end
end

if saveMP4
    close(vw);
end

fprintf('动画已保存：%s（GIF），%s（MP4）\n', gifName, mp4Name);


