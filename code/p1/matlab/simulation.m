%% simulate_problem1_anim.m
% 题目A - 问题1：FY1 投放 1 枚烟幕干扰弹干扰 M1 的仿真动画（中文注释）
% 功能：
% 1) 三维动画：展示导弹M1、FY1运动，释放/起爆点，烟幕云团（球）及其下沉，假目标(原点)与真目标圆柱。
% 2) 动画右上角实时标注：时间t、云团中心到 M1->真目标 视线段的距离 d(t)、是否处于遮蔽(半径10m内且在线段上)。
% 3) 收尾生成 d(t) 曲线图，画出阈值10m、起爆/失效时刻、遮蔽时间区间；并在命令行报告总遮蔽时长。
% 4) 动画保存：problem1_anim.gif 与 problem1_anim.mp4

clear; clc; close all;

%% ===================== 常量与场景参数 =====================
g        = 9.80665;          % 重力加速度 (m/s^2)
v_sink   = 3.0;          % 烟幕云团起爆后下沉速度 (m/s)
R_cloud  = 10.0;         % 有效遮蔽半径 (m)
T_eff    = 20.0;         % 起爆后有效时间 (s)

% 假目标（原点）与真目标（圆柱，下底圆心在(0,200,0)，半径7m，高10m）
decoy = [0, 0, 0];
real_center = [0, 200, 0];
real_r = 7.0;
real_h = 10.0;
% 判据使用真目标“代表点”（圆柱中心高度），可改为 [0,200,0] 或 [0,200,10] 做敏感性
Tpt = [0, 200, 5];

% 导弹 M1：速度 300 m/s，方向指向原点
M0 = [20000, 0, 2000];
uM = -M0 / norm(M0);
vM = 300 * uM;

% 无人机 FY1：以 120 m/s、等高度直线飞向原点（沿 -x 方向）
FY1_0   = [17800, 0, 1800];
vU      = 120;
heading = [-1, 0, 0]; heading = heading / norm(heading);

% 时间：1.5 s 投放；再过 3.6 s 起爆
t_release = 1.5;
t_fuze    = 3.6;
t_exp     = t_release + t_fuze;   % = 5.1 s

% 释放/起爆位置（水平跟随无人机，竖直为抛体落差）
P_rel = FY1_0 + vU * t_release * heading;                 % 释放点
E_xy  = FY1_0 + vU * (t_release + t_fuze) * heading;      % 起爆水平位置
x_e   = E_xy(1); y_e = E_xy(2);
z_e   = FY1_0(3) - 0.5 * g * (t_fuze^2);
Ept   = [x_e, y_e, z_e];

%% ===================== 仿真时间轴 =====================
t0   = 0.0;
t1   = t_exp + T_eff;   % 模拟至云团失效
dt   = 0.02;            % 动画步长（越小越细腻）
t    = (t0:dt:t1).';    % 列向量

N = numel(t);

%% ===================== 轨迹预计算（便于统一设轴范围与加速绘图） =====================
% 导弹轨迹
M = M0 + t .* vM;             % N x 3
% 无人机轨迹（等高度）
U = FY1_0 + t .* (vU * heading);
U(:,3) = FY1_0(3);                          

% 烟幕云团中心（仅在 [t_exp, t_exp+T_eff] 有效）
C = nan(N,3);
idxCloud = (t >= t_exp) & (t <= t_exp + T_eff);
C(idxCloud,1) = x_e;
C(idxCloud,2) = y_e;
C(idxCloud,3) = z_e - v_sink * (t(idxCloud) - t_exp);

%% ===================== 遮蔽判据（云团中心到 M(t)->Tpt 线段的最短距离） =====================
d    = nan(N,1);   % 距离
tau  = nan(N,1);   % 线段参数（未截断）
mask = false(N,1); % 是否处于遮蔽（d<=R_cloud 且 0<=tau<=1）

for k = find(idxCloud).'
    Ck = C(k,:);
    Mk = M(k,:);
    [dk, tk] = distPointToSegment(Ck, Mk, Tpt);  % 见文末局部函数
    d(k)     = dk;
    tau(k)   = tk;
    mask(k)  = (dk <= R_cloud) && (tk >= 0) && (tk <= 1);
end

% 遮蔽时间段（用于曲线图高亮）
edges = diff([false; mask; false]);
idxStart = find(edges == 1);
idxEnd   = find(edges == -1) - 1;
occlusion_total = sum(mask) * dt;

fprintf('【遮蔽总时长】 %.3f s\n', occlusion_total);
if ~isempty(idxStart)
    fprintf('【遮蔽起止】 [%.3f s, %.3f s]\n', t(idxStart(1)), t(idxEnd(end)));
else
    fprintf('当前参数下无遮蔽窗口。\n');
end

%% ===================== 3D 动画绘制设置 =====================
% 为了视觉稳定，预先确定统一坐标轴范围
allXYZ = [M; U; decoy; Ept; [real_center(1), real_center(2), real_h]];
xlim3 = [min(allXYZ(:,1))-1500, max(allXYZ(:,1))+1500];
ylim3 = [min(allXYZ(:,2))-1500, max(allXYZ(:,2))+1500];
zlim3 = [0, max(allXYZ(:,3))+500];

% 真目标圆柱网格（用于半透明展示）
[th, zz] = meshgrid(linspace(0,2*pi,60), linspace(0,real_h,8));
xc = real_center(1) + real_r * cos(th);
yc = real_center(2) + real_r * sin(th);
zc = zz;

% 预计算单位球用于快速更新烟幕球
[su, sv] = sphere(24);  % 分辨率适中
% 初始化图窗
fig = figure('Color','w','Position',[80 80 980 640]);
set(fig, 'Resize','off');  % 锁定窗口尺寸，避免帧大小在动画中变化
ax  = axes(fig); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
axis(ax,'equal');
xlabel(ax,'X (m)'); ylabel(ax,'Y (m)'); zlabel(ax,'Z (m)');
title(ax, '问题1：FY1-单弹干扰M1（三维动画）');
xlim(ax, xlim3); ylim(ax, ylim3); zlim(ax, zlim3);
view(ax, 30, 20);

% 静态要素：真目标圆柱、诱饵、释放/起爆点（仅符号，不加近点注释，图例已标注）
surf(ax, xc, yc, zc, 'EdgeAlpha',0.25,'FaceAlpha',0.15,'FaceColor',[0.3 0.6 1.0]);
plot3(ax, decoy(1),decoy(2),decoy(3), 'x',  'LineWidth',1.2,'MarkerSize',9);
plot3(ax, P_rel(1),P_rel(2),P_rel(3), 'ko','MarkerSize',8,'LineWidth',1.0);
plot3(ax, Ept(1), Ept(2), Ept(3),    'ks','MarkerSize',9,'LineWidth',1.0);

% 动态要素：导弹/无人机轨迹、当前位置、LOS、云团球
hMpath = plot3(ax, nan, nan, nan, '-', 'LineWidth',1.2);
hUpath = plot3(ax, nan, nan, nan, '-', 'LineWidth',1.2);
hMnow  = plot3(ax, nan, nan, nan, '^', 'MarkerSize',8, 'MarkerFaceColor','auto');
hUnow  = plot3(ax, nan, nan, nan, 'o', 'MarkerSize',8, 'MarkerFaceColor','auto');
hLOS   = plot3(ax, nan, nan, nan, '-.', 'LineWidth',1.0);
hCloud = surf(ax, nan(2), nan(2), nan(2), 'EdgeAlpha',0.2, 'FaceAlpha',0.25);
% 视锥（点光源=导弹，对应被烟幕遮挡的阴影锥体）
hCone  = surf(ax, nan(2), nan(2), nan(2), 'EdgeAlpha',0.15, 'FaceAlpha',0.15, 'FaceColor',[1.0 0.5 0.5]);
% 云团球心到 LOS 的垂线与数值标注
hPerp  = plot3(ax, nan, nan, nan, ':', 'LineWidth',1.0, 'Color',[0.4 0.4 0.4]);
hDText = text(ax, 0,0,0, '', 'FontSize',9, 'Color',[0.2 0.2 0.2], ...
    'HorizontalAlignment','center','VerticalAlignment','bottom','Visible','off');
% 右上角文本：时间t、d(t)、遮蔽状态
txtInfo = text(ax, xlim3(2), ylim3(2), zlim3(2), '', ...
    'HorizontalAlignment','right','VerticalAlignment','top','FontSize',10,'FontWeight','bold');

legend(ax, {'真目标(圆柱)','诱饵','释放点','起爆点','M1轨迹','FY1轨迹','M1当前位置','FY1当前位置','LOS(M1->真目标)','烟幕球'}, ...
    'Location','northeastoutside','FontSize',9);

% 动画输出设置
saveGIF = true;
saveMP4 = true;
gifName = 'output/problem1_anim.gif';
mp4Name = 'output/problem1_anim.mp4';
fps     = 25;

if saveMP4
    vw = VideoWriter(mp4Name, 'MPEG-4');
    vw.FrameRate = fps;
    open(vw);
end

%% ===================== 动画主循环 =====================
% 为提升效率，可步进若干帧保存一次
step = 1;   % 每步1帧（可改成2/3以减小文件大小）

% 统一视频帧尺寸（在首帧获得并固定，且保证偶数，以满足H.264）
targetH = [];
targetW = [];
for k = 1:step:N
    % 轨迹到当前
    set(hMpath, 'XData', M(1:k,1), 'YData', M(1:k,2), 'ZData', M(1:k,3));
    set(hUpath, 'XData', U(1:k,1), 'YData', U(1:k,2), 'ZData', U(1:k,3));
    set(hMnow,  'XData', M(k,1),   'YData', M(k,2),   'ZData', M(k,3));
    set(hUnow,  'XData', U(k,1),   'YData', U(k,2),   'ZData', U(k,3));

    % LOS（导弹当前位置 -> 真目标代表点）
    set(hLOS, 'XData', [M(k,1), Tpt(1)], 'YData', [M(k,2), Tpt(2)], 'ZData', [M(k,3), Tpt(3)]);

    % 云团球（若已起爆则更新位置；否则隐藏）
    if idxCloud(k)
        Ck = C(k,:);
        Xs = R_cloud * su + Ck(1);
        Ys = R_cloud * sv + Ck(2);
        Zs = R_cloud * su * 0 + R_cloud * sv * 0; % 先占位，避免 set surf 三元组不一致
        % 用 set 更新更稳：需要直接设置 'XData','YData','ZData'
        set(hCloud, 'XData', R_cloud*su + Ck(1), ...
                    'YData', R_cloud*sv + Ck(2), ...
                    'ZData', R_cloud*su*0 + Ck(3) + R_cloud*sv*0); % 等效球面平移
        set(hCloud, 'Visible','on');
        % LOS 可见，并根据遮蔽状态着色（遮蔽=蓝色虚线；未遮蔽=红色实线）
        set(hLOS, 'Visible','on');
        if mask(k)
            set(hLOS, 'Color',[0 0.45 0.74], 'LineStyle','--');
        else
            set(hLOS, 'Color',[1 0 0], 'LineStyle','-');
        end

        % 更新云团球心到 LOS 的垂线与数值
        Mk = M(k,:);
        AB = Tpt - Mk;
        denomAB = dot(AB,AB);
        if denomAB > eps
            tauk = dot(Ck - Mk, AB) / denomAB;
            tclamp = min(max(tauk,0),1);
            Q = Mk + tclamp * AB;
            set(hPerp, 'XData',[Ck(1) Q(1)], 'YData',[Ck(2) Q(2)], 'ZData',[Ck(3) Q(3)], 'Visible','on');
            midp = (Ck + Q)/2;
            set(hDText, 'Position', midp, 'String', sprintf('d=%.2f m', d(k)), 'Visible','on');
            if mask(k)
                set(hPerp,  'Color',[0 0.45 0.74]);
                set(hDText, 'Color',[0 0.45 0.74]);
            else
                set(hPerp,  'Color',[1 0 0]);
                set(hDText, 'Color',[1 0 0]);
            end
        else
            set(hPerp, 'Visible','off');
            set(hDText,'Visible','off');
        end

        % ========== 视锥更新（导弹作为点光源）==========
        % 视锥定义：所有从导弹当前位置 M(k,:) 出发，刚好与球体切触的光线所张成的圆锥
        Mk = M(k,:);
        vCM = Ck - Mk;                 % 光心到球心向量
        L   = norm(vCM);
        if (L > R_cloud + 1e-6)
            % 锥半角：sin(alpha) = R / L
            alpha = asin(min(1, R_cloud / L));
            % 锥轴为 e3
            e3 = vCM / L;
            % 在与 e3 垂直平面取正交基 e1,e2
            tmp = [0 0 1]; if abs(dot(tmp,e3)) > 0.9, tmp = [0 1 0]; end
            e1 = cross(e3, tmp); e1 = e1 / norm(e1);
            e2 = cross(e3, e1);
            % 取一圈圆周方向参数 phi
            phi = linspace(0, 2*pi, 60);
            % 生成锥侧面的两层（近端/远端）用于 surf
            rdir = cos(alpha)*e3 + sin(alpha)*(cos(phi').*e1 + sin(phi').*e2);
            % 取两个尺度系数，确保覆盖到圆柱/场景边界（更长一些）
            s_near = 0.0;                 % 顶点处
            sceneSpan = max([diff(xlim3), diff(ylim3), diff(zlim3)]);
            s_far  = L + 2.0*sceneSpan;   % 远端长度：与场景跨度相关，显著延伸
            P0 = Mk + s_near * rdir;      % 60x3
            P1 = Mk + s_far  * rdir;      % 60x3
            % 转为 2x60 网格给 surf
            Xc = [P0(:,1)'; P1(:,1)'];
            Yc = [P0(:,2)'; P1(:,2)'];
            Zc = [P0(:,3)'; P1(:,3)'];
            set(hCone, 'XData', Xc, 'YData', Yc, 'ZData', Zc, 'Visible','on');
        else
            set(hCone, 'Visible','off');
        end
    else
        set(hCloud, 'Visible','off');
        set(hCone,  'Visible','off');
        % 云团未生成/已失效：LOS 仍显示为红色实线
        set(hLOS, 'XData', [M(k,1), Tpt(1)], 'YData', [M(k,2), Tpt(2)], 'ZData', [M(k,3), Tpt(3)], ...
                  'Visible','on', 'Color',[1 0 0], 'LineStyle','-');
        set(hPerp,  'Visible','off');
        set(hDText, 'Visible','off');
    end

    % 信息文本
    tt   = t(k);
    dstr = 'N/A';
    occ  = '否';
    if ~isnan(d(k))
        dstr = sprintf('%.2f m', d(k));
        if mask(k), occ = '是'; end
    end
    cum_occ = sum(mask(1:k)) * dt;  % 累计遮蔽时长（实时）
    txt = sprintf('t = %.2f s\n云团球心距LOS: %s\n遮蔽中: %s\n累计遮蔽: %.2f s', tt, dstr, occ, cum_occ);
    set(txtInfo, 'String', txt);

    drawnow;

    % 采帧并统一尺寸
    fr = getframe(fig);
    img = fr.cdata;  % uint8 HxWx3
    [h,w,~] = size(img);
    if isempty(targetH)
        % 首帧确定目标尺寸，并强制为偶数
        targetH = h - mod(h,2);
        targetW = w - mod(w,2);
        if targetH <= 0, targetH = h; end
        if targetW <= 0, targetW = w; end
    end
    % 裁剪/填充到目标尺寸（避免使用额外工具箱）
    Ht = targetH; Wt = targetW;
    Hc = h;       Wc = w;
    % 初始化目标画布（黑边填充）
    out = zeros(Ht, Wt, 3, 'uint8');
    hcopy = min(Ht, Hc);
    wcopy = min(Wt, Wc);
    out(1:hcopy, 1:wcopy, :) = img(1:hcopy, 1:wcopy, :);

    % 保存 GIF 帧（使用索引色减少体积）
    if saveGIF
        [A,map] = rgb2ind(out, 256);
        if k == 1
            imwrite(A, map, gifName, 'gif', 'LoopCount', Inf, 'DelayTime', 1/fps);
        else
            imwrite(A, map, gifName, 'gif', 'WriteMode', 'append', 'DelayTime', 1/fps);
        end
    end
    % 保存 MP4 帧（尺寸固定，满足 H.264 要求）
    if saveMP4
        writeVideo(vw, out);
    end
end

if saveMP4
    close(vw);
end

fprintf('动画已保存：%s（GIF），%s（MP4）\n', gifName, mp4Name);

%% ===================== d(t) 曲线与遮蔽窗口 =====================
figure('Color','w','Position',[80 760 1180 360]); hold on; grid on;
plot(t, d, 'LineWidth',1.4);
yline(R_cloud,'--','R=10 m');
xline(t_exp,'-.','起爆');
xline(t_exp+T_eff,':','失效');

% 阴影高亮遮蔽区间
yTop = max(R_cloud*2, max(d(~isnan(d)))*1.05);
for i = 1:numel(idxStart)
    xs = t(idxStart(i)); xe = t(idxEnd(i));
    patch([xs xe xe xs], [0 0 yTop yTop], [0.8 0.9 1.0], 'FaceAlpha',0.35, 'EdgeColor','none');
end
xlabel('时间 t (s)');
ylabel('云团中心到视线段的距离 d(t) (m)');
title(sprintf('遮蔽总时长：%.3f s', occlusion_total));
xlim([t0, t1]); ylim([0, yTop]);

% 在坐标轴上标注“开始/结束遮蔽”的时间
for i = 1:numel(idxStart)
    xs = t(idxStart(i));
    xe = t(idxEnd(i));
    xline(xs,'--',sprintf('开始 %.3f s', xs), ...
        'LabelVerticalAlignment','bottom','LabelHorizontalAlignment','left', ...
        'Color',[0 0.5 0],'LineWidth',1.0);
    xline(xe,':',sprintf('结束 %.3f s', xe), ...
        'LabelVerticalAlignment','bottom','LabelHorizontalAlignment','left', ...
        'Color',[0.5 0 0.5],'LineWidth',1.0);
end

%% ===================== 终端结果打印（用于表格） =====================
% 方向角：以 +x 轴为0°，逆时针为正，范围 [0,360)
dir_deg = mod(atan2d(heading(2), heading(1)), 360);
smoke_id = 1;  % 单弹情形编号固定为1（如有多弹可自行修改）

fprintf('\n======= 结果 =======\n');
fprintf('无人机运动方向(度) : %.2f\n', dir_deg);
fprintf('无人机运动速度(m/s) : %.2f\n', vU);
fprintf('烟幕干扰弹编号 : %d\n', smoke_id);
fprintf('烟幕干扰弹投放点的x坐标(m) : %.3f\n', P_rel(1));
fprintf('烟幕干扰弹投放点的y坐标(m) : %.3f\n', P_rel(2));
fprintf('烟幕干扰弹投放点的z坐标(m) : %.3f\n', P_rel(3));
fprintf('烟幕干扰弹起爆点的x坐标(m) : %.3f\n', Ept(1));
fprintf('烟幕干扰弹起爆点的y坐标(m) : %.3f\n', Ept(2));
fprintf('烟幕干扰弹起爆点的z坐标(m) : %.3f\n', Ept(3));
fprintf('有效干扰时长(s) : %.3f\n', occlusion_total);

% 遮蔽起止区间
if exist('idxStart','var') && exist('idxEnd','var') && ~isempty(idxStart)
    fprintf('【遮蔽起止】 ');
    for i = 1:numel(idxStart)
        xs = t(idxStart(i));
        xe = t(idxEnd(i));
        if i == 1
            fprintf('[%.3f s, %.3f s]', xs, xe);
        else
            fprintf(', [%.3f s, %.3f s]', xs, xe);
        end
    end
    fprintf('\n');
end

%% ===================== 局部函数：点到线段距离 =====================
function [dist, tau] = distPointToSegment(C, A, B)
% distPointToSegment
% 计算点C到线段A-B的最短距离，并返回参数tau（未截断）
% tau<0 最近点在A延长线上；tau>1 最近点在B延长线上；0<=tau<=1 在段内。
    AB = B - A;
    denom = dot(AB, AB);
    if denom < eps
        tau = 0;
        dist = norm(C - A);
        return;
    end
    tau = dot(C - A, AB) / denom;
    tclamp = min(max(tau, 0), 1);
    P = A + tclamp * AB;
    dist = norm(C - P);
end

